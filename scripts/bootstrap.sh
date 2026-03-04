#!/usr/bin/env bash
set -euo pipefail

echo "=== SayFit Bootstrap ==="

# 1. Create venv & install
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]

# 2. Env file
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example – please set OPENAI_API_KEY."
fi

# 3. Extract food databases
if [ ! -f data/processed/usda_nutrition_clean.csv ]; then
  echo "Extracting food databases …"
  unzip -o data/food_dbs.zip -d data/processed/
  rm -rf data/processed/__MACOSX
fi

echo ""
echo "Bootstrap complete!"
echo "  1. Edit .env and set OPENAI_API_KEY"
echo "  2. Run: sayfit ingest   (builds the FAISS index)"
echo "  3. Run: sayfit run 'I ate 3 eggs and a banana'"

