#!/usr/bin/env python3
"""
SayFit 🎙️🥗 – Voice-to-Nutrition (Terminal Mode)

Records audio from your microphone, transcribes with Whisper,
then runs the full SayFit nutrition pipeline.

Usage:
    python voice_recorder.py          # Record → Transcribe → Analyze
    python voice_recorder.py --text   # Skip recording, just type what you ate
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

import whisper
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

load_dotenv()

# ---------- CONFIG ----------
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_SECONDS = 15
OUTDIR = Path("data")
OUTPUT_DIR = Path("outputs")
WHISPER_MODEL = "small"
LANGUAGE = None  # None = auto-detect
TARGET_PEAK_DBFS = -1.0
# -----------------------------

console = Console()


def ensure_dirs():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_audio(audio_i16: np.ndarray, target_dbfs: float = -1.0):
    audio_f = audio_i16.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(audio_f))) if audio_f.size else 0.0
    if peak <= 0.0:
        return audio_i16
    target_peak = 10 ** (target_dbfs / 20.0)
    gain = target_peak / peak
    audio_f = np.clip(audio_f * gain, -1.0, 1.0)
    return (audio_f * 32767.0).astype(np.int16)


def record_audio() -> np.ndarray | None:
    """Record audio from microphone. Press Enter to stop early."""
    console.print()
    console.print(Panel(
        f"[bold green]🎤 Recording[/bold green] (max {MAX_SECONDS}s)\n"
        "Speak what you ate, then [bold]press Enter[/bold] to stop.",
        title="SayFit Voice Input",
    ))

    frames: list[np.ndarray] = []
    recording = True

    def callback(indata, frame_count, time_info, status):
        if recording:
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=callback,
    )

    try:
        stream.start()
        input()  # Block until Enter is pressed
    except KeyboardInterrupt:
        pass
    finally:
        recording = False
        stream.stop()
        stream.close()

    if not frames:
        return None

    return np.concatenate(frames, axis=0)


def transcribe(audio: np.ndarray) -> str:
    """Save audio, normalize, and transcribe with Whisper."""
    rec_id = uuid.uuid4().hex[:12]
    wav_path = OUTDIR / f"audio_{rec_id}.wav"
    wav_write(str(wav_path), SAMPLE_RATE, audio)

    # Normalize
    audio_norm = normalize_audio(audio, TARGET_PEAK_DBFS)
    wav_norm_path = OUTDIR / f"audio_{rec_id}_norm.wav"
    wav_write(str(wav_norm_path), SAMPLE_RATE, audio_norm)

    console.print("[dim]Loading Whisper model …[/dim]")
    model = whisper.load_model(WHISPER_MODEL)

    console.print("[dim]Transcribing …[/dim]")
    result = model.transcribe(
        str(wav_norm_path),
        language=LANGUAGE,
        fp16=False,
        temperature=0.0,
        beam_size=5,
    )
    text = (result.get("text") or "").strip()

    # Save transcript JSON
    json_path = OUTDIR / f"transcript_{rec_id}.json"
    payload = {
        "id": rec_id,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "text": text,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[dim]Transcript saved → {json_path}[/dim]")

    return text


def run_nutrition_pipeline(text: str, timestamp: str | None = None):
    """Run the SayFit nutrition pipeline on transcribed text."""
    from sayfit.config import get_settings
    from sayfit.vector_store import FoodVectorStore
    from sayfit.pipeline import SayFitPipeline

    settings = get_settings()
    index_dir = settings.faiss_index_dir

    if not (index_dir / FoodVectorStore.INDEX_FILE).exists():
        console.print("[bold red]ERROR:[/bold red] No FAISS index found. "
                      "Run [bold]sayfit ingest[/bold] first.")
        sys.exit(1)

    store = FoodVectorStore.load(index_dir)
    pipe = SayFitPipeline(settings=settings, store=store, console=console)

    rec_id = uuid.uuid4().hex[:12]
    save_path = OUTPUT_DIR / f"voice_{rec_id}.json"

    ts = timestamp or dt.datetime.now().isoformat(timespec="seconds")
    pipe.run(text, timestamp=ts, save_path=save_path)


def main():
    parser = argparse.ArgumentParser(description="SayFit Voice-to-Nutrition")
    parser.add_argument("--text", action="store_true",
                        help="Skip recording – type what you ate instead")
    args = parser.parse_args()

    ensure_dirs()

    console.print()
    console.print("[bold cyan]SayFit 🎙️🥗 – Voice-to-Nutrition[/bold cyan]")
    console.print()

    while True:
        if args.text:
            text = console.input("[bold green]Type what you ate:[/bold green] ").strip()
            if not text:
                console.print("[yellow]Empty input, try again.[/yellow]")
                continue
        else:
            audio = record_audio()
            if audio is None or len(audio) < SAMPLE_RATE * 0.5:  # < 0.5s
                console.print("[yellow]Recording too short, try again.[/yellow]")
                continue

            text = transcribe(audio)
            if not text:
                console.print("[yellow]No speech detected, try again.[/yellow]")
                continue

        console.print(f'\n[bold]📝 Recognized:[/bold] "{text}"\n')

        # Run the nutrition pipeline
        run_nutrition_pipeline(text)

        console.print()
        if not Confirm.ask("Record another meal?", default=True):
            break

    console.print("[bold cyan]Goodbye! 👋[/bold cyan]")


if __name__ == "__main__":
    main()
