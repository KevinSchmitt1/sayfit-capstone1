# SayFit 🎙️🥗 – Voice-to-Nutrition Pipeline

Speak what you ate. Get macros back.

SayFit is a **RAG-powered nutrition tracking system** that turns natural language food descriptions into structured macro-nutrient breakdowns using real food databases (USDA + OpenFoodFacts).

---

## Architecture Overview

```
Voice / Text Input
       │
       ▼
┌──────────────────────────────┐
│  Step 0: ASR (Whisper)       │  voice_gui_recorder.py (optional)
│  "I ate 3 eggs and a pizza"  │  → {"text": "..."}
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 1: Item Extraction     │  OpenAI LLM with strict JSON schema
│  → [{name, amount, unit}]    │  src/sayfit/extraction.py
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 2: RAG Retrieval       │  FAISS vector search (sentence-transformers)
│  → top-K candidates per item │  src/sayfit/retrieval.py + vector_store.py
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 3: Selection/Grounding │  Deterministic rerank (or LLM rerank)
│  → best match from candidates│  src/sayfit/selection.py
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 4: Portion Norm + Math │  Unit conversion + portion defaults
│  → grams + macros per item   │  src/sayfit/portion.py
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Step 5: Output              │  Rich table + JSON
│  → items + totals            │  src/sayfit/output.py
└──────────────────────────────┘
```

## Project Structure

```
.
├── src/sayfit/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point (sayfit ingest|run|file|batch)
│   ├── config.py           # Pydantic settings from .env
│   ├── models.py           # Pydantic data models
│   ├── extraction.py       # Step 1: LLM item extraction (OpenAI)
│   ├── retrieval.py        # Step 2: RAG retrieval
│   ├── vector_store.py     # FAISS index + sentence-transformers
│   ├── selection.py        # Step 3: candidate selection / reranking
│   ├── portion.py          # Step 4: portion normalization + nutrition math
│   ├── output.py           # Step 5: output formatting (rich table + JSON)
│   └── pipeline.py         # End-to-end pipeline orchestrator
├── src/sayfit/ingestion.py  # Data loading (USDA + OFF CSVs → unified index)
├── data/
│   ├── food_dbs.zip        # Source data (USDA + OpenFoodFacts)
│   ├── processed/          # Extracted CSVs
│   └── vector_store/       # Persisted FAISS index
├── input_samples/           # Example input JSON files
├── outputs/                 # Pipeline output files
├── voice_gui_recorder.py    # Optional: Whisper-based voice recorder GUI
├── tests/
├── .env.example
└── pyproject.toml
```

## Quickstart

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Extract food databases

```bash
unzip data/food_dbs.zip -d data/processed/
```

### 4. Build the FAISS vector index

```bash
sayfit ingest
```

This embeds ~1.8M USDA + ~66K OpenFoodFacts food entries using `all-MiniLM-L6-v2` and builds a FAISS index.

### 5. Run the pipeline

**Single text:**
```bash
sayfit run "I ate 3 eggs and a pepperoni pizza"
```

**Single JSON file:**
```bash
sayfit file input_samples/test1.json
```

**Batch (all test files):**
```bash
sayfit batch input_samples/
```

**With LLM-based reranking:**
```bash
sayfit run "100g oatmeal, 1 banana, 200ml almond milk" --llm-rerank
```

## Example Output

```
🍽  Meal Nutrition Breakdown
┌──────────────┬───────────────┬────────┬───────┬──────────┬─────────────┬───────────┬──────────┐
│ Food         │ Matched       │ Source │ Grams │ Calories │ Protein (g) │ Carbs (g) │ Fat (g)  │
├──────────────┼───────────────┼────────┼───────┼──────────┼─────────────┼───────────┼──────────┤
│ oatmeal      │ OATMEAL       │ usda   │   100 │    389.0 │        16.9 │      66.3 │      6.9 │
│ banana       │ BANANA        │ usda   │   120 │    106.8 │         1.3 │      27.1 │      0.4 │
│ almond milk  │ ALMOND MILK   │ off    │   200 │     30.0 │         1.2 │       5.4 │      1.2 │
├──────────────┼───────────────┼────────┼───────┼──────────┼─────────────┼───────────┼──────────┤
│ TOTAL        │               │        │       │    525.8 │        19.4 │      98.8 │      8.5 │
└──────────────┴───────────────┴────────┴───────┴──────────┴─────────────┴───────────┴──────────┘
```

## How RAG Works in This System

Unlike traditional RAG (retrieving text passages), SayFit retrieves **structured food rows** from nutrition databases:

1. **Knowledge base** = USDA + OpenFoodFacts food entries (name + macros per 100g)
2. **Embeddings** = `all-MiniLM-L6-v2` via sentence-transformers
3. **Vector index** = FAISS (flat inner product on normalized vectors)
4. **Grounding** = The LLM can only select from retrieved candidates, never invent macros

## Tech Stack

| Component           | Technology                          |
|---------------------|-------------------------------------|
| LLM                 | OpenAI (gpt-5.2)                    |
| Embeddings          | sentence-transformers (all-MiniLM)  |
| Vector Search       | FAISS (faiss-cpu)                   |
| Data                | USDA + OpenFoodFacts (CSV)          |
| Schemas             | Pydantic v2                         |
| CLI Output          | Rich                                |
| ASR (optional)      | OpenAI Whisper                      |

## Tests

```bash
pytest
```

## Audio Recording (Optional)

If you want to record voice input and transcribe with Whisper:

```bash
pip install -e .[audio]
python voice_gui_recorder.py
```

This produces a JSON file in `data/` with `{"text": "..."}` that can be fed to the pipeline.

