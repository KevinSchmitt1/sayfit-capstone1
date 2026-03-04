# SayFit Nutrition Pipeline

End-to-end pipeline for turning meal text into grounded nutrition totals using USDA + OpenFoodFacts data.

Implemented flow (from `structure.md`):

1. Input JSON (`text`, optional `timestamp`)
2. Item extraction (`name`, `amount`, `unit`) with optional Groq LLM and deterministic fallback
3. Retrieval (RAG over structured food rows) from USDA + OFF with top K=8 candidates
4. Grounded selection (deterministic rerank with USDA/OFF preference logic)
5. Portion normalization to grams (portion table + fallback lexicon)
6. Deterministic macro math
7. Output with per-item details, totals, follow-ups, and coaching tips

## Data sources

The app uses these files in `data/processed/`:

- `off_nutrition_clean.csv`
- `usda_nutrition_clean.csv`

If missing, it auto-extracts them from `data/food_dbs.zip`.

## Run

## Quickstart (fresh clone)

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install all Python dependencies (single source of truth):

```bash
pip install -U pip
pip install -r requirements.txt
```

3. System dependency for Whisper:

- Install `ffmpeg` and ensure it is in your PATH.
- macOS example:

```bash
brew install ffmpeg
```

4. Optional audio backend note (macOS/Linux):

- If `sounddevice` fails to open input devices, install PortAudio.
- macOS example:

```bash
brew install portaudio
```

## Run

Single file:

```bash
python main.py --input input_samples/test1.json --output-dir outputs --data-dir data
```

Batch mode (all `input_samples/*.json`):

```bash
python main.py --output-dir outputs --data-dir data
```

Optional goal context:

```bash
python main.py --input input_samples/test1.json --goal "fat loss"
```

## Voice recorder GUI (record → transcribe → final JSON)

Run:

```bash
python voice_gui_recorder.py
```

What it does:

1. Records microphone audio to WAV
2. Transcribes with Whisper
3. Saves raw transcript JSON to `data/transcript_<id>.json`
4. Runs nutrition pipeline on transcript text
5. Saves app-ready output JSON to `outputs/final_audio_<id>.json`

The final JSON has the same schema as CLI output (`items`, `totals`, `follow_ups`, `coaching`).

## Output schema

Each output JSON includes:

- `items`: extracted + matched foods with grams, source, confidence, assumptions, nutrition
- `totals`: calories/protein/carbs/fat
- `follow_ups`: clarification prompts for low-confidence matches
- `coaching`: simple end-of-day suggestions

## Optional Groq extraction

If `GROQ_API_KEY` is set in `.env`, extraction step attempts LLM parsing first.
If unavailable or failing, deterministic extraction is used automatically.

## Notes on requirements files

- `requirements.txt` is the canonical dependency list for this repo.
- `requirements_audio.txt` and `requirements_data.txt` are compatibility wrappers that forward to `requirements.txt`.
