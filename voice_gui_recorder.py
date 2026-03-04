import threading
import datetime as dt
import uuid
import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

import whisper

sys.path.insert(0, str((Path(__file__).resolve().parent / "src")))
from sayfit_pipeline import NutritionPipeline


# ---------- CONFIG ----------
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_SECONDS = 15

OUTDIR = Path("data")
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")

# Accuracy Hebel:
# tiny < base < small < medium < large
WHISPER_MODEL = "small"
WHISPER_FALLBACK_MODEL = "tiny"
LANGUAGE = None  # "de", "en", or None for auto-detect

# Normalisierung: Ziel-Peak in dBFS (z.B. -1 dBFS)
TARGET_PEAK_DBFS = -1.0
SAVE_NORMALIZED_WAV = True  # nur zur Kontrolle/Debug; Transkription nutzt die normierte Datei
# ---------------------------


def iso_now_local() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dirs():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_int16_to_target_peak(audio_i16: np.ndarray, target_peak_dbfs: float = -1.0):
    """
    audio_i16: int16 array shape (N,1) or (N,)
    Returns: normalized int16 array (same shape), applied_gain_db (float), peak_before_dbfs (float)
    """
    if audio_i16.dtype != np.int16:
        audio_i16 = audio_i16.astype(np.int16)

    # flatten peak calc but keep original shape
    audio_f = audio_i16.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(audio_f))) if audio_f.size else 0.0

    if peak <= 0.0:
        return audio_i16, 0.0, float("-inf")

    peak_before_dbfs = 20.0 * np.log10(peak)

    target_peak = 10 ** (target_peak_dbfs / 20.0)
    gain = target_peak / peak
    gain_db = 20.0 * np.log10(gain)

    audio_f = np.clip(audio_f * gain, -1.0, 1.0)
    audio_norm_i16 = (audio_f * 32767.0).astype(np.int16)

    return audio_norm_i16, gain_db, peak_before_dbfs


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Voice Recorder → Whisper (JSON Output)")
        root.geometry("760x380")
        root.resizable(False, False)

        ensure_dirs()

        self.is_recording = False
        self.stream = None
        self.frames = []
        self.auto_stop_job = None

        self.whisper_model = None
        self.pipeline = None
        self.model_lock = threading.Lock()
        self.pipeline_lock = threading.Lock()

        self.status_var = tk.StringVar(value="Ready. Click Start (Space toggles).")
        self.btn = ttk.Button(root, text="Start Recording", command=self.toggle)
        self.btn.pack(pady=14)

        self.status_lbl = ttk.Label(root, textvariable=self.status_var, wraplength=720)
        self.status_lbl.pack(pady=6)

        self.text_box = tk.Text(root, height=12, width=95)
        self.text_box.pack(padx=12, pady=10)
        self._set_text("Transcript will appear here after recording.\n")

        root.bind("<space>", lambda _e: self.toggle())

    def toggle(self):
        if not self.is_recording:
            self.start()
        else:
            self.stop(manual=True)

    def start(self):
        self.frames = []
        self.is_recording = True
        self.btn.config(text="Stop Recording")
        self.status_var.set(f"Recording… (auto-stop in {MAX_SECONDS}s).")

        self.auto_stop_job = self.root.after(MAX_SECONDS * 1000, lambda: self.stop(manual=False))

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as e:
            self._reset_state()
            messagebox.showerror("Recording Error", f"Could not start recording:\n{e}")

    def _audio_callback(self, indata, frames, time_info, status):
        self.frames.append(indata.copy())

    def stop(self, manual: bool):
        if not self.is_recording:
            return

        self.status_var.set("Stopping recording…")
        self.btn.config(state="disabled")

        if manual and self.auto_stop_job is not None:
            try:
                self.root.after_cancel(self.auto_stop_job)
            except Exception:
                pass
            self.auto_stop_job = None

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass

        self.stream = None
        self.is_recording = False

        if not self.frames:
            self._reset_state()
            self.status_var.set("No audio captured. Ready.")
            self.btn.config(state="normal", text="Start Recording")
            return

        audio_i16 = np.concatenate(self.frames, axis=0)  # shape (N,1), int16

        created_at = iso_now_local()
        rec_id = uuid.uuid4().hex

        wav_path = OUTDIR / f"audio_{rec_id}.wav"
        wav_norm_path = OUTDIR / f"audio_{rec_id}_norm.wav"
        json_path = OUTDIR / f"transcript_{rec_id}.json"

        # Save original WAV
        wav_write(str(wav_path), SAMPLE_RATE, audio_i16)

        # Normalize for better ASR
        audio_norm_i16, gain_db, peak_before_dbfs = normalize_int16_to_target_peak(
            audio_i16, TARGET_PEAK_DBFS
        )

        if SAVE_NORMALIZED_WAV:
            wav_write(str(wav_norm_path), SAMPLE_RATE, audio_norm_i16)

        self.status_var.set(
            f"Saved audio. Normalized (gain {gain_db:+.1f} dB, peak was {peak_before_dbfs:.1f} dBFS). Transcribing…"
        )
        self._set_text("Transcribing… please wait.\n")

        # Transcribe in background from in-memory normalized audio.
        # This avoids ffmpeg decoding dependency for file-path based transcription.
        audio_for_asr = audio_norm_i16.reshape(-1).astype(np.float32) / 32768.0
        threading.Thread(
            target=self._transcribe_and_save,
            args=(audio_for_asr, json_path, rec_id, created_at, wav_path, wav_norm_path),
            daemon=True,
        ).start()

    def _load_whisper(self):
        with self.model_lock:
            if self.whisper_model is None:
                try:
                    self.whisper_model = whisper.load_model(WHISPER_MODEL)
                except Exception:
                    self.whisper_model = whisper.load_model(WHISPER_FALLBACK_MODEL)
            return self.whisper_model

    def _load_pipeline(self):
        with self.pipeline_lock:
            if self.pipeline is None:
                self.pipeline = NutritionPipeline(data_dir=DATA_DIR, top_k=8)
            return self.pipeline

    def _transcribe_and_save(
        self,
        audio_for_asr: np.ndarray,
        json_path: Path,
        rec_id: str,
        created_at: str,
        wav_original: Path,
        wav_norm: Path,
    ):
        try:
            model = self._load_whisper()

            # Stabilere Settings (weniger "Raten")
            result = model.transcribe(
                audio_for_asr,
                language=LANGUAGE,
                fp16=False,          # CPU-safe
                temperature=0.0,     # deterministischer
                beam_size=5,         # etwas bessere Decoding-Qualität
            )
            text = (result.get("text") or "").strip()

            payload = {
                "id": rec_id,
                "created_at": created_at,
                "timestamp": created_at,
                "text": text,
            }
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            final_path = OUTPUT_DIR / f"final_audio_{rec_id}.json"
            pipeline = self._load_pipeline()
            pipeline_result = pipeline.run(payload)
            pipeline_out = pipeline.result_to_dict(pipeline_result)
            final_path.write_text(json.dumps(pipeline_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            self.root.after(0, lambda: self._done(text, wav_original, wav_norm, json_path, final_path))
        except Exception as e:
            self.root.after(0, lambda: self._err(e))

    def _done(self, text: str, wav_path: Path, wav_norm_path: Path, json_path: Path, final_path: Path):
        self.btn.config(state="normal", text="Start Recording")
        if SAVE_NORMALIZED_WAV:
            self.status_var.set(
                f"Done. Saved: {wav_path.name}, {wav_norm_path.name}, {json_path.name}, {final_path.name}"
            )
        else:
            self.status_var.set(f"Done. Saved: {wav_path.name}, {json_path.name}, {final_path.name}")
        self._set_text(text if text else "(No speech recognized)")

    def _err(self, e: Exception):
        self.btn.config(state="normal", text="Start Recording")
        self.status_var.set("Error during transcription.")
        msg = str(e)
        hints = []
        low = msg.lower()
        if "ffmpeg" in low:
            hints.append("Install ffmpeg and ensure it is available in PATH.")
        if "model" in low and "load" in low:
            hints.append("Check internet access for first model download or switch to a smaller model.")
        if "no module named" in low:
            hints.append("Reinstall dependencies with: pip install -r requirements.txt")

        detail = msg
        if hints:
            detail += "\n\nPossible fixes:\n- " + "\n- ".join(hints)

        messagebox.showerror("Transcription Error", detail)
        self._set_text("")

    def _reset_state(self):
        self.is_recording = False
        self.auto_stop_job = None
        self.frames = []

    def _set_text(self, text: str):
        self.text_box.config(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text + "\n")
        self.text_box.config(state="disabled")


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("aqua")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
