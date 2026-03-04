Architecture overview
0) Input (already done)

ASR → JSON like:

{"text": "i ate a pepperoni pizza and 3 eggs"}
1) Item extraction (LLM, NOT RAG yet)

Goal: turn raw text into a structured list:

[
  {"query":"pepperoni pizza", "quantity":1, "unit":"serving"},
  {"query":"egg", "quantity":3, "unit":"piece"}
]

Use an instruction LLM (Mistral/Qwen/Phi) with a strict JSON schema.

This is a parsing problem.

2) Retrieval (this is your “RAG”)

For each extracted item:

Retrieve candidate foods from your databases:

USDA (generic foods)

OpenFoodFacts (branded/packaged foods)

Use embeddings + vector search to get top K candidates (e.g., 10–30).

Optionally combine with lexical filtering (SQL ILIKE in DuckDB) for better precision.

3) Selection / grounding (small LLM step or deterministic rerank)

Now choose the best match from candidates.
Two options:

Deterministic: pick highest similarity, prefer USDA unless query looks branded.

LLM rerank: show LLM the candidates and ask it to pick the closest one (still grounded because it must choose from the list).

4) Portion normalization + math (deterministic)

Convert units to grams:

3 eggs → estimate grams (e.g., 1 large egg ~ 50g edible portion) using a small portion table

1 serving pizza → if database has per 100g, assume default serving grams or ask user
Then compute:

totals = Σ (grams/100 × macros_per_100g)

5) Output

Return:

table of items + matched food name + grams + macros

daily totals

6) End-of-day coaching (LLM)

Input:

totals + goal
Output:

improvements + substitutions

What “RAG” means in this system

Your “knowledge base” is not documents — it’s structured food rows.

So your RAG retrieves candidate food rows (name + macros per 100g), not text passages.

You ground the LLM by only allowing it to:

output structured items (step 1)

select from retrieved candidates (step 3)
and never invent macros.

Recommended tool stack (simple and robust)
Core

Python backend

DuckDB stores your cleaned tables (USDA + OFF)

Embeddings (HF) -> vector index

FAISS (local vector search) or Chroma

LLM inference (HF Transformers)

Orchestration

You can use LangChain/LlamaIndex, but you don’t need them.
For a capstone, I’d keep it minimal:

plain Python functions + Pydantic schemas

add LangChain later if you want “framework credibility”

My recommendation: plain Python + FAISS + DuckDB.

How to integrate your data into RAG
Build an index table like this (one row per food entry)

For retrieval you need:

doc_id (unique)

source (“usda” or “off”)

name (food_name)

brand (OFF only)

macros (kcal/protein/carbs/fat per 100g)

text_for_embedding:

USDA: "description"

OFF: "product_name | brands | categories_en"

Then embed text_for_embedding and put vectors into FAISS/Chroma.

Handling “pizza” correctly (important realism)

“pepperoni pizza” is ambiguous. Your system should support uncertainty:

If the best match confidence is low:

ask follow-up: “Was it homemade, restaurant, or frozen packaged? Approx grams/slices?”

Or use a default assumption:

“1 serving pepperoni pizza = 150g” (store this in portion defaults)

For a capstone, it’s totally acceptable to:

implement defaults + allow user corrections.

Minimal pipeline: what each component does
LLM #1 (Extraction)

Input: raw text
Output: JSON items (query, quantity, unit)

Retriever (RAG)

Input: item query text
Output: top K candidate food rows from USDA/OFF

LLM #2 (optional reranker)

Input: item + candidates
Output: chosen candidate ID + grams assumption if needed

Deterministic calculator

Input: chosen rows + portion grams
Output: macros per item + totals

LLM #3 (coaching)

Input: totals + goal
Output: tips/substitutions