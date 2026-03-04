from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os

sys.path.insert(0, str((Path(__file__).resolve().parent / "src")))
from sayfit_pipeline import NutritionPipeline


def load_env_file(path: Path = Path(".env")) -> None:
    """Load .env values into os.environ (without overwriting existing env vars)."""
    if not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SayFit nutrition pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to input JSON with {'text': ..., 'timestamp': ...}",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input_samples"),
        help="Directory of input JSON files (used when --input is omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory containing processed CSVs or food_dbs.zip",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Optional daily goal context (e.g. fat loss, muscle gain)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of retrieval candidates before selection (default: 20)",
    )
    return parser


def run_single(pipeline: NutritionPipeline, input_file: Path, output_dir: Path, goal: str | None) -> Path:
    payload = json.loads(input_file.read_text())
    result = pipeline.run(payload, goal=goal)
    output = pipeline.result_to_dict(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"final_{input_file.name}"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


def main() -> None:
    load_env_file(Path(".env"))
    args = build_parser().parse_args()
    pipeline = NutritionPipeline(data_dir=args.data_dir, top_k=args.top_k)

    if args.input:
        out_path = run_single(pipeline, args.input, args.output_dir, args.goal)
        print(f"Pipeline finished. Output written to {out_path}")
        return

    input_files = sorted(args.input_dir.glob("*.json"))
    if not input_files:
        raise FileNotFoundError(f"No json files found in {args.input_dir}")

    for input_file in input_files:
        out_path = run_single(pipeline, input_file, args.output_dir, args.goal)
        print(f"Pipeline finished. Output written to {out_path}")


if __name__ == "__main__":
    main()
