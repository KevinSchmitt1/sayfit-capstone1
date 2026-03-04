"""
SayFit CLI – main entry point.

Usage:
  sayfit ingest          Build the FAISS index from USDA + OFF CSVs.
  sayfit run  "<text>"   Run the full pipeline on a text string.
  sayfit file <path>     Run the pipeline on a JSON file ({"text": "...", "timestamp": "..."}).
  sayfit batch <dir>     Run the pipeline on every JSON file in a directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from sayfit.config import get_settings


def cmd_ingest(args: argparse.Namespace) -> None:
    """Build the FAISS vector index from CSV data."""
    from sayfit.ingestion import build_food_index
    from sayfit.vector_store import FoodVectorStore

    s = get_settings()
    console = Console()

    console.print("[bold]Building food index …[/bold]")
    df = build_food_index(s.usda_csv, s.off_csv)

    # Optional: limit rows for faster dev/testing
    if args.max_rows and args.max_rows < len(df):
        console.print(f"[yellow]Sampling {args.max_rows:,} rows (--max-rows)[/yellow]")
        df = df.sample(n=args.max_rows, random_state=42).reset_index(drop=True)

    console.print("[bold]Creating FAISS vector store …[/bold]")
    store = FoodVectorStore.build_from_dataframe(df, model_name=s.embedding_model)
    store.save(s.faiss_index_dir)
    console.print("[bold green]✓ Index built successfully.[/bold green]")


def _load_store():
    """Load an existing FAISS store."""
    from sayfit.vector_store import FoodVectorStore
    s = get_settings()
    index_dir = s.faiss_index_dir
    if not (index_dir / FoodVectorStore.INDEX_FILE).exists():
        print(f"ERROR: No FAISS index found at {index_dir}. Run 'sayfit ingest' first.",
              file=sys.stderr)
        sys.exit(1)
    return FoodVectorStore.load(index_dir)


def _make_pipeline(store, *, use_llm_rerank: bool = False):
    from sayfit.pipeline import SayFitPipeline
    return SayFitPipeline(
        settings=get_settings(),
        store=store,
        use_llm_rerank=use_llm_rerank,
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Run pipeline on a single text string."""
    store = _load_store()
    pipe = _make_pipeline(store, use_llm_rerank=args.llm_rerank)
    pipe.run(args.text, save_path=Path(args.output) if args.output else None)


def cmd_file(args: argparse.Namespace) -> None:
    """Run pipeline on a JSON file."""
    path = Path(args.path)
    data = json.loads(path.read_text(encoding="utf-8"))
    text = data["text"]
    timestamp = data.get("timestamp")

    store = _load_store()
    pipe = _make_pipeline(store, use_llm_rerank=args.llm_rerank)

    out_name = f"result_{path.stem}.json"
    out_path = Path(args.output_dir) / out_name

    pipe.run(text, timestamp=timestamp, save_path=out_path)


def cmd_batch(args: argparse.Namespace) -> None:
    """Run pipeline on all JSON files in a directory."""
    input_dir = Path(args.dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob("*.json"))

    if not files:
        print(f"No JSON files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    store = _load_store()
    pipe = _make_pipeline(store, use_llm_rerank=args.llm_rerank)

    console = Console()
    console.print(f"[bold]Processing {len(files)} file(s) …[/bold]\n")

    for f in files:
        console.rule(f"[bold]{f.name}[/bold]")
        data = json.loads(f.read_text(encoding="utf-8"))
        text = data["text"]
        timestamp = data.get("timestamp")
        out_path = output_dir / f"result_{f.stem}.json"
        pipe.run(text, timestamp=timestamp, save_path=out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sayfit",
        description="SayFit – Voice-to-nutrition pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_p = subparsers.add_parser("ingest", help="Build FAISS index from USDA + OFF CSVs")
    ingest_p.add_argument(
        "--max-rows", type=int, default=None,
        help="Limit total rows (random sample) for faster dev builds",
    )

    # run
    run_p = subparsers.add_parser("run", help="Run pipeline on a text string")
    run_p.add_argument("text", help="What you ate, in natural language")
    run_p.add_argument("-o", "--output", default=None, help="Save result JSON to this path")
    run_p.add_argument("--llm-rerank", action="store_true", help="Use LLM for candidate reranking")

    # file
    file_p = subparsers.add_parser("file", help="Run pipeline on a JSON file")
    file_p.add_argument("path", help="Path to input JSON file")
    file_p.add_argument("--output-dir", default="outputs", help="Directory for result files")
    file_p.add_argument("--llm-rerank", action="store_true", help="Use LLM for candidate reranking")

    # batch
    batch_p = subparsers.add_parser("batch", help="Run pipeline on all JSON files in a directory")
    batch_p.add_argument("dir", help="Directory containing input JSON files")
    batch_p.add_argument("--output-dir", default="outputs", help="Directory for result files")
    batch_p.add_argument("--llm-rerank", action="store_true", help="Use LLM for candidate reranking")

    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    handlers = {
        "ingest": cmd_ingest,
        "run": cmd_run,
        "file": cmd_file,
        "batch": cmd_batch,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
