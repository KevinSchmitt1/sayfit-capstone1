import argparse
import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timezone

DEFAULT_ROOT = Path.home() / "Documents" / "superwhisper"
DEFAULT_DB = Path("transcripts.db")


DESIRED_COLUMNS = [
    # name, type
    ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("source", "TEXT"),
    ("json_path", "TEXT"),  # we enforce uniqueness via a UNIQUE INDEX (more migration-friendly)
    ("created_at", "TEXT"),
    ("file_mtime_utc", "TEXT"),
    ("model_name", "TEXT"),
    ("app_version", "TEXT"),
    ("duration_ms", "INTEGER"),
    ("processing_time_ms", "INTEGER"),
    ("text", "TEXT"),
]


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;", (name,)
    ).fetchone()
    return row is not None


def get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    # rows: (cid, name, type, notnull, dflt_value, pk)
    return {r[1] for r in rows}


def ensure_schema(db_path: Path) -> None:
    """
    Makes the DB schema forward-compatible:
    - creates missing tables
    - adds missing columns
    - ensures indexes exist
    - backfills legacy rows so json_path exists (for dedupe)
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")  # safer for watcher use
        conn.execute("PRAGMA foreign_keys=ON;")

        # Optional meta table (nice to have)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )

        if not table_exists(conn, "transcripts"):
            # fresh create
            cols_sql = ",\n".join([f"{name} {typ}" for name, typ in DESIRED_COLUMNS])
            conn.execute(f"CREATE TABLE transcripts (\n{cols_sql}\n);")
            conn.commit()
        else:
            # table exists: add missing columns (migration-safe)
            existing = get_table_columns(conn, "transcripts")

            for name, typ in DESIRED_COLUMNS:
                if name in existing:
                    continue
                # id primary key cannot be added; but if table lacks id, it likely has rowid anyway.
                if name == "id":
                    continue
                conn.execute(f"ALTER TABLE transcripts ADD COLUMN {name} {typ};")

            conn.commit()

        # Ensure indexes (created even if table existed before)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_text ON transcripts(text);")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_json_path ON transcripts(json_path);")
        conn.commit()

        # Backfill legacy rows (only affects rows that were inserted before this schema existed)
        cols = get_table_columns(conn, "transcripts")

        if "source" in cols:
            conn.execute(
                "UPDATE transcripts SET source='legacy' WHERE source IS NULL OR source='';"
            )

        if "json_path" in cols:
            # Make sure existing rows have unique json_path values (for dedupe)
            # Use id if available, else rowid.
            if "id" in cols:
                conn.execute(
                    """
                    UPDATE transcripts
                    SET json_path = 'legacy:' || id
                    WHERE json_path IS NULL OR json_path='';
                    """
                )
            else:
                conn.execute(
                    """
                    UPDATE transcripts
                    SET json_path = 'legacy:' || rowid
                    WHERE json_path IS NULL OR json_path='';
                    """
                )

        # Mark schema version
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', '1');"
        )
        conn.commit()


def stable_read_json(path: Path, tries: int = 6, wait_s: float = 0.15):
    """
    Superwhisper may still be writing the file; check size stability before reading.
    """
    last = None
    for _ in range(tries):
        try:
            size = path.stat().st_size
            if last is not None and size == last:
                txt = path.read_text(encoding="utf-8")
                return json.loads(txt)
            last = size
        except Exception:
            pass
        time.sleep(wait_s)
    return None


def extract_text(obj) -> str:
    """
    Heuristic transcript extraction:
    tries common keys and nested structures.
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, (int, float, bool)):
        return ""
    if isinstance(obj, list):
        parts = [extract_text(x) for x in obj]
        parts = [p for p in parts if p]
        return " ".join(parts).strip()
    if isinstance(obj, dict):
        for k in ("result", "text", "transcript", "finalText", "final_text", "output"):
            if k in obj:
                t = extract_text(obj.get(k))
                if t:
                    return t

        for k in ("data", "payload", "response", "results", "content"):
            if k in obj:
                t = extract_text(obj.get(k))
                if t:
                    return t

        for v in obj.values():
            t = extract_text(v)
            if t:
                return t
    return ""


def extract_meta(obj: dict) -> dict:
    if not isinstance(obj, dict):
        return {}

    def pick(*keys):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return None

    created = pick("datetime", "created_at", "createdAt", "timestamp", "time")
    model = pick("modelName", "model_name", "model")
    appv = pick("appVersion", "app_version")
    duration = pick("duration", "durationMs", "duration_ms")
    proc = pick("processingTime", "processing_time", "processingTimeMs", "processing_time_ms")

    created_iso = None
    if isinstance(created, str) and created:
        s = created.replace("Z", "+00:00")
        try:
            d = datetime.fromisoformat(s)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            created_iso = d.astimezone(timezone.utc).isoformat()
        except Exception:
            created_iso = created
    elif isinstance(created, (int, float)):
        try:
            created_iso = datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
        except Exception:
            created_iso = None

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return None

    return {
        "created_at": created_iso,
        "model_name": str(model) if model else None,
        "app_version": str(appv) if appv else None,
        "duration_ms": to_int(duration),
        "processing_time_ms": to_int(proc),
    }


def already_ingested(conn: sqlite3.Connection, json_path: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM transcripts WHERE json_path = ? LIMIT 1;", (json_path,)
    )
    return cur.fetchone() is not None


def iter_json_files(root: Path):
    for p in root.rglob("*.json"):
        if any(part.startswith(".") for part in p.parts):
            continue
        yield p


def ingest_once(root: Path, db_path: Path, min_chars: int = 1) -> int:
    ensure_schema(db_path)

    if not root.exists():
        raise FileNotFoundError(f"Superwhisper folder not found: {root}")

    ingested = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")

        for json_file in sorted(iter_json_files(root), key=lambda p: p.stat().st_mtime):
            jp = str(json_file)

            if already_ingested(conn, jp):
                continue

            data = stable_read_json(json_file)
            if data is None:
                continue

            text = extract_text(data)
            if len(text) < min_chars:
                continue

            meta = extract_meta(data)

            conn.execute(
                """
                INSERT INTO transcripts (
                    source, json_path, created_at, file_mtime_utc,
                    model_name, app_version, duration_ms, processing_time_ms, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "superwhisper",
                    jp,
                    meta.get("created_at"),
                    mtime_utc(json_file),
                    meta.get("model_name"),
                    meta.get("app_version"),
                    meta.get("duration_ms"),
                    meta.get("processing_time_ms"),
                    text,
                ),
            )
            conn.commit()
            ingested += 1
            print(f"[OK] {json_file.name}: {text[:120]}{'...' if len(text) > 120 else ''}")

    return ingested


def search_db(db_path: Path, keyword: str, limit: int = 20):
    ensure_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT created_at, file_mtime_utc, json_path, text
            FROM transcripts
            WHERE text LIKE ?
            ORDER BY id DESC
            LIMIT ?;
            """,
            (f"%{keyword}%", limit),
        ).fetchall()

    if not rows:
        print("[SEARCH] No matches.")
        return

    for created_at, mtime, json_path, text in rows:
        print("\n---")
        print("created_at:", created_at or "(unknown)")
        print("file_mtime:", mtime)
        print("json_path:", json_path)
        print("text:", text)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Superwhisper JSON transcripts into SQLite (for later keyword search)."
    )
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="Superwhisper folder root.")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="SQLite db path.")
    parser.add_argument("--once", action="store_true", help="Run one ingestion pass and exit.")
    parser.add_argument("--watch", action="store_true", help="Continuously watch (polling).")
    parser.add_argument("--poll", type=float, default=1.0, help="Polling interval in seconds (watch mode).")
    parser.add_argument("--search", type=str, default=None, help="Search keyword in DB and exit.")
    parser.add_argument("--limit", type=int, default=20, help="Search result limit.")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    db_path = Path(args.db).expanduser()

    if args.search:
        search_db(db_path, args.search, limit=args.limit)
        return

    if not args.once and not args.watch:
        args.watch = True

    if args.once:
        n = ingest_once(root, db_path)
        print(f"[DONE] ingested={n}")
        return

    print(f"[WATCH] root={root}")
    print(f"[WATCH] db={db_path}")
    print("[WATCH] Press Ctrl+C to stop.")
    while True:
        try:
            n = ingest_once(root, db_path)
            if n:
                print(f"[WATCH] ingested={n} @ {iso_utc_now()}")
            time.sleep(args.poll)
        except KeyboardInterrupt:
            print("\n[STOP] bye.")
            break


if __name__ == "__main__":
    main()