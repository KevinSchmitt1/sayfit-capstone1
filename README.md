# Python RAG Project Template

This template gives you a clean baseline for building a Retrieval-Augmented Generation (RAG) system with an LLM.

## Project structure

```
.
├── data/
│   ├── raw/              # Put source docs here
│   ├── processed/        # Optional intermediate outputs
│   └── vector_store/     # Persisted embeddings index
├── notebooks/            # Experiments
├── scripts/              # Helper scripts
├── src/rag_template/
│   ├── app.py            # CLI: ingest + query
│   ├── config/           # Settings and env config
│   ├── core/             # Shared types
│   ├── generation/       # Prompting + answer synthesis
│   ├── ingestion/        # Loaders + chunking
│   ├── pipeline/         # End-to-end indexing/query flows
│   ├── providers/        # LLM + embedding providers
│   ├── retrieval/        # Vector store + retrieval
│   └── utils/            # Utilities
├── tests/
├── .env.example
└── pyproject.toml
```

## Quickstart

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

2. Configure env vars:

```bash
cp .env .env
# fill OPENAI_API_KEY
```

3. Add text/markdown files in `data/raw/`.

4. Build index:

```bash
python -m rag_template.app ingest --data-dir data/raw
```

5. Ask questions:

```bash
python -m rag_template.app query "What are the key points in the documents?"
```

## Notes

- Vector store is a local JSON-backed numpy implementation for simplicity.
- Replace provider classes in `providers/` to use other LLMs or embedding APIs.
- Extend retrieval (reranking, metadata filters, hybrid search) as needed.
