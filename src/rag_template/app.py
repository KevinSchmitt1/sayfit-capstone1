import argparse
from pathlib import Path

from dotenv import load_dotenv

from rag_template.config.settings import get_settings
from rag_template.pipeline.indexing import build_index
from rag_template.pipeline.rag_chain import answer_query
from rag_template.providers.openai_provider import OpenAIChatProvider, OpenAIEmbeddingProvider


DEFAULT_INDEX_PATH = Path("data/vector_store/index.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG template CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Load docs and build vector index")
    ingest.add_argument("--data-dir", default="data/raw", help="Directory of source documents")
    ingest.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Output index path")

    query = subparsers.add_parser("query", help="Ask a question against indexed docs")
    query.add_argument("question", help="User question")
    query.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Existing index path")

    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    settings = get_settings()

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to .env or your environment.")

    embedding_provider = OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    chat_provider = OpenAIChatProvider(
        api_key=settings.openai_api_key,
        model=settings.model_name,
    )

    if args.command == "ingest":
        num_chunks = build_index(
            data_dir=Path(args.data_dir),
            out_path=Path(args.index_path),
            settings=settings,
            embeddings=embedding_provider,
        )
        print(f"Index built successfully: {num_chunks} chunks -> {args.index_path}")
        return

    if args.command == "query":
        result = answer_query(
            query=args.question,
            index_path=Path(args.index_path),
            settings=settings,
            embeddings=embedding_provider,
            chat=chat_provider,
        )
        print(result)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
