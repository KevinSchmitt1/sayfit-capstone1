# Architecture Overview

## 0. Input (already done)

ASR output JSON:

voice_gui_recorder.py: use this for transcribing and parsing .wav to .json

- includes a normalizer and a db-adjuster

```json
{"text": "i ate a pepperoni pizza and 3 eggs"}
```

## 1. Item Extraction (LLM, not RAG yet)
Input: json from step 0.

Goal: turn raw text (json) into a structured list (json):

How to handle missing data: if "amount" or "unit" is missing, keep it missing for now. Will be processed after retrieving the data from the food databases.

__Example:__ first output
```json
{
  "items": [
    {
      "name": "oatmeal",
      "amount": 100.0,
      "unit": "g"
    },
    {
      "name": "apple",
      "amount": 150.0,
      "unit": "g"
    },
    {
      "name": "almond milk",
      "amount": 200.0,
      "unit": "ml"
    }
  ],
  "timestamp": "2026-03-04T08:00:00"
}
```

Use an instruction LLM (groq) with a strict JSON schema.

## 2. Retrieval (your RAG)

For each extracted item, retrieve candidate foods from your databases:

- USDA (generic foods)
- OpenFoodFacts (branded/packaged foods)

The databases are in the food_dbs.zip and will be extracted to 'off_nutrition_clean.csv', 'usda_nutrition_clean.csv' -> use these first for the RAG.
For default unzipping use: 'data/processed/...' as path.

Use embeddings + vector search to get top K candidates (8).


## 3. Selection / Grounding (small LLM step or deterministic rerank)

Choose the best match from candidates.


API key to use from .env

Two options:

- Deterministic: pick highest similarity, prefer USDA unless query looks branded.
- LLM rerank: show LLM the candidates and ask it to pick the closest one (still grounded because it must choose from the list).

Built in some sanity checks/verification steps if the foods are accurate enough for the user.

## 4. Portion Normalization + Math (deterministic)

Convert (missing) units to grams:

- `3 eggs` -> estimate grams (e.g., `1 large egg ~ 50g edible portion`) using a small portion table.
- `1 serving pizza` -> if database has per 100g, assume default serving grams or ask user.
- `noodles` -> use reasonable portion size (recommendations by health organizations or similar)

Build a lexikon for adjusting serving sizes/ give examples for unknown serving sizes like 'hand full', etc.

Then compute. Use the same API key as before (or is it advisable to use a different one?)

This is a parsing problem. First output to second output:

__Example:__
```
{
  "items": [
    {
      "name": "oatmeal",
      "amount": 100.0,
      "unit": "g",
      "matched_name": "oatmeal",
      "nutrition": {
        "calories": 389.0,
        "protein": 16.9,
        "fiber": 10.6
      }
    },
    {
      "name": "apple",
      "amount": 150.0,
      "unit": "g",
      "matched_name": "apple",
      "nutrition": {
        "calories": 78.0,
        "protein": 0.45,
        "fiber": 3.6
      }
    },
    {
      "name": "almond milk",
      "amount": 200.0,
      "unit": "ml",
      "matched_name": "almond milk",
      "nutrition": {
        "calories": 30.0,
        "protein": 1.2,
        "fiber": 0.4
      }
    }
  ],
  "timestamp": "2026-03-04T08:00:00",
  "totals": {
    "calories": 497.0,
    "protein": 18.549999999999997,
    "fiber": 14.6
  }
}
```

## 5. Output

Return:

- Table of items + matched food name + grams + macros
- Daily totals


## What "RAG" Means in This System

Your knowledge base is not documents; it is structured food rows.

So your RAG retrieves candidate food rows (`name + macros per 100g`), not text passages.

You ground the LLM by only allowing it to:

- Output structured items (step 1)
- Select from retrieved candidates (step 3)
- Never invent macros, if something is unknown, predict/assume the macros/kcal but ask the user/tell them that its unknown

## Recommended Tool Stack (simple and robust)

### Core

- Python backend
- Cleaned data as .csv (if possible)
- Embeddings (HF) -> vector index
- FAISS (local vector search) or Chroma
- LLM inference (HF Transformers)

## How to Integrate Data into RAG

Build an index table with one row per food entry.

For retrieval you need:

- `doc_id` (unique)
- `source` (`"usda"` or `"off"`)
- `name` (`food_name`)
- `brand` (OFF only)
- `macros` (`kcal/protein/carbs/fat` per 100g)
- `text_for_embedding`

Examples:

- USDA: `"description"`
- OFF: `"product_name | brands | categories_en"`

Then embed `text_for_embedding` and store vectors in FAISS/Chroma.

## Handling "pizza" Correctly (important realism)

`"pepperoni pizza"` is ambiguous. Support uncertainty:

If best-match confidence is low:

- Ask follow-up: "Was it homemade, restaurant, or frozen packaged? Approx grams/slices?"

Or use a default assumption:

- `1 serving pepperoni pizza = 150g` (store this in portion defaults)

For a capstone, it's acceptable to implement defaults and allow user corrections.

## Minimal Pipeline: Component Responsibilities

### Transcription tool #0
- Input audio (as .wav) convert to json.
- OpenAIwhisper + sounddevice

### LLM #1 (Extraction)

- Input: raw text
- Output: JSON items ('name', 'amount', 'unit')

### Retriever (RAG)

- Input: item query text
- Output: top K candidate food rows from USDA/OFF

### LLM #2 (optional reranker)

- Input: item + candidates
- Output: chosen candidate ID + grams assumption if needed

### Deterministic Calculator

- Input: chosen rows + portion grams
- Output: macros per item + totals

### LLM #3 (coaching)

- Input: totals + goal
- Output: tips/substitutions
