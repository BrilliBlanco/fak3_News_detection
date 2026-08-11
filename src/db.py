"""
Prediction store - the "data management" half of the project.

Every classification the CLI or the app makes can be logged to a local SQLite
file (`reports/predictions.db`). That turns a one-shot demo into something you
can actually analyse afterwards:

  - what the model sees in practice vs. what it was trained on
  - confidence distribution on real user input (usually far shakier than on
    the test split)
  - which inputs users flagged as wrong -> an exportable correction set that
    can be folded back into training

SQLite because it is in the Python standard library: no server, no install,
one file that can be copied between teammates or deleted to reset.

Usage:
    python src/db.py --stats
    python src/db.py --recent 20
    python src/db.py --export reports/predictions.csv
    python src/db.py --export-corrections data/corrections.csv
    python src/db.py --feedback 12 --correct-label fake
    python src/db.py --clear
"""

import argparse
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL,
    source       TEXT    NOT NULL,   -- text | url | image | csv | cli
    origin       TEXT,               -- the URL / filename, when there is one
    model        TEXT    NOT NULL,   -- registry key: nb | svm | lr
    input_hash   TEXT    NOT NULL,   -- sha256 of the raw input, for dedup
    text_preview TEXT,
    char_count   INTEGER,
    word_count   INTEGER,
    prediction   INTEGER NOT NULL,   -- 0 fake, 1 real
    label        TEXT    NOT NULL,
    confidence   REAL,
    proba_fake   REAL,
    proba_real   REAL,
    top_terms    TEXT,               -- JSON list of [term, contribution]
    user_feedback TEXT,              -- NULL | correct | incorrect
    true_label   INTEGER             -- set when a user supplies the right answer
);
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_pred_model   ON predictions(model);
CREATE INDEX IF NOT EXISTS idx_pred_hash    ON predictions(input_hash);
"""


@contextmanager
def connect(db_path=None):
    """Context-managed connection that always commits and closes."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def log_prediction(text: str, result: dict, model_key: str, source: str = "cli",
                   origin: str = None, top_terms=None, db_path=None) -> int:
    """
    Persist one prediction. `result` is the dict returned by
    `explain.explain_prediction` (or anything with the same keys).
    Returns the new row id.
    """
    init_db(db_path)
    proba = result.get("proba") or {}
    terms_json = json.dumps(top_terms[:10]) if top_terms else None

    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (created_at, source, origin, model, input_hash, text_preview,
                char_count, word_count, prediction, label, confidence,
                proba_fake, proba_real, top_terms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                source, origin, model_key, _hash(text), text[:500],
                len(text), len(text.split()),
                int(result["prediction"]), result["label"],
                result.get("confidence"), proba.get(0), proba.get(1), terms_json,
            ),
        )
        return cur.lastrowid


def fetch_predictions(limit: int = 100, model: str = None, label: str = None,
                      since: str = None, db_path=None) -> pd.DataFrame:
    """Recent predictions as a dataframe, newest first."""
    init_db(db_path)
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    if model:
        query += " AND model = ?"
        params.append(model)
    if label:
        query += " AND label = ?"
        params.append(label.upper())
    if since:
        query += " AND created_at >= ?"
        params.append(since)
    query += " ORDER BY id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def summary_stats(db_path=None) -> dict:
    """Aggregates for the dashboard / the --stats CLI."""
    init_db(db_path)
    with connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
        if not total:
            return {"total": 0}

        by_label = {r["label"]: r["c"] for r in conn.execute(
            "SELECT label, COUNT(*) c FROM predictions GROUP BY label")}
        by_model = {r["model"]: r["c"] for r in conn.execute(
            "SELECT model, COUNT(*) c FROM predictions GROUP BY model")}
        by_source = {r["source"]: r["c"] for r in conn.execute(
            "SELECT source, COUNT(*) c FROM predictions GROUP BY source")}
        avg_conf = conn.execute(
            "SELECT AVG(confidence) a FROM predictions").fetchone()["a"]
        low_conf = conn.execute(
            "SELECT COUNT(*) c FROM predictions WHERE confidence < 0.7").fetchone()["c"]
        feedback = {r["user_feedback"] or "none": r["c"] for r in conn.execute(
            "SELECT user_feedback, COUNT(*) c FROM predictions GROUP BY user_feedback")}
        repeats = conn.execute(
            "SELECT COUNT(*) c FROM (SELECT input_hash FROM predictions "
            "GROUP BY input_hash HAVING COUNT(*) > 1)").fetchone()["c"]
        first = conn.execute("SELECT MIN(created_at) m FROM predictions").fetchone()["m"]

    reviewed = feedback.get("correct", 0) + feedback.get("incorrect", 0)
    return {
        "total": total,
        "by_label": by_label,
        "by_model": by_model,
        "by_source": by_source,
        "avg_confidence": avg_conf,
        "low_confidence_count": low_conf,
        "low_confidence_share": low_conf / total,
        "feedback": feedback,
        "reviewed": reviewed,
        "observed_accuracy": (feedback.get("correct", 0) / reviewed) if reviewed else None,
        "repeated_inputs": repeats,
        "first_logged": first,
    }


def record_feedback(row_id: int, correct: bool = None, true_label: int = None,
                    db_path=None) -> bool:
    """
    Attach a human verdict to a logged prediction. Either say whether the
    prediction was right (`correct`), or give the actual label (`true_label`),
    from which correctness is derived.
    """
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT prediction FROM predictions WHERE id = ?",
                           (row_id,)).fetchone()
        if row is None:
            return False
        if true_label is not None:
            correct = (int(true_label) == row["prediction"])
        conn.execute(
            "UPDATE predictions SET user_feedback = ?, true_label = ? WHERE id = ?",
            ("correct" if correct else "incorrect",
             true_label if true_label is not None else
             (row["prediction"] if correct else 1 - row["prediction"]),
             row_id),
        )
    return True


def export_csv(path, db_path=None) -> int:
    """Dump the whole log. Returns the row count written."""
    df = fetch_predictions(limit=0, db_path=db_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


def export_corrections(path, db_path=None) -> int:
    """
    Export human-labelled rows as a training-ready CSV (text, label).

    This is the feedback loop: run the app, have someone review the flagged
    items, export here, and you have new in-domain labelled data that came
    from the distribution the model is actually used on.
    """
    init_db(db_path)
    with connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT text_preview AS text, true_label AS label, model, created_at "
            "FROM predictions WHERE true_label IS NOT NULL", conn)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


def clear(db_path=None) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        n = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
        conn.execute("DELETE FROM predictions")
    return n


# --- CLI -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Inspect and manage the prediction log.")
    p.add_argument("--db", default=None, help=f"SQLite file (default {DB_PATH})")
    p.add_argument("--stats", action="store_true", help="Print aggregate statistics")
    p.add_argument("--recent", type=int, metavar="N", help="Show the N most recent predictions")
    p.add_argument("--model", help="Filter by model key (nb/svm/lr)")
    p.add_argument("--export", metavar="PATH", help="Write the full log to CSV")
    p.add_argument("--export-corrections", metavar="PATH",
                   help="Write human-labelled rows as a training-ready CSV")
    p.add_argument("--feedback", type=int, metavar="ID", help="Row id to annotate")
    p.add_argument("--correct-label", choices=["fake", "real"],
                   help="The true label for the --feedback row")
    p.add_argument("--clear", action="store_true", help="Delete every logged prediction")
    return p.parse_args()


def main():
    args = parse_args()

    if args.clear:
        confirm = input(f"Delete every row in {args.db or DB_PATH}? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return
        print(f"Deleted {clear(args.db)} rows.")
        return

    if args.feedback is not None:
        if not args.correct_label:
            print("--feedback also needs --correct-label fake|real")
            return
        true_label = 0 if args.correct_label == "fake" else 1
        ok = record_feedback(args.feedback, true_label=true_label, db_path=args.db)
        print(f"Row {args.feedback}: recorded true label {args.correct_label}."
              if ok else f"No prediction with id {args.feedback}.")
        return

    if args.export:
        print(f"Wrote {export_csv(args.export, args.db)} rows to {args.export}")
        return

    if args.export_corrections:
        n = export_corrections(args.export_corrections, args.db)
        print(f"Wrote {n} human-labelled rows to {args.export_corrections}")
        if not n:
            print("(none yet - use --feedback ID --correct-label fake|real, "
                  "or the Review tab in the app)")
        return

    if args.recent:
        df = fetch_predictions(limit=args.recent, model=args.model, db_path=args.db)
        if df.empty:
            print("No predictions logged yet.")
            return
        cols = ["id", "created_at", "source", "model", "label", "confidence", "text_preview"]
        df = df[cols].copy()
        df["text_preview"] = df["text_preview"].str.slice(0, 55)
        df["confidence"] = df["confidence"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
        print(df.to_string(index=False))
        return

    # default: --stats
    s = summary_stats(args.db)
    if not s["total"]:
        print("No predictions logged yet. Run a prediction with --log first:")
        print('  python src/predict.py --text "..." --log')
        return

    print(f"Prediction log: {args.db or DB_PATH}")
    print(f"  total predictions : {s['total']:,}  (since {s['first_logged']})")
    print(f"  by label          : {s['by_label']}")
    print(f"  by model          : {s['by_model']}")
    print(f"  by input source   : {s['by_source']}")
    print(f"  mean confidence   : {s['avg_confidence']:.1%}")
    print(f"  low confidence    : {s['low_confidence_count']} "
          f"({s['low_confidence_share']:.1%} below 70%)")
    print(f"  repeated inputs   : {s['repeated_inputs']}")
    print(f"  human-reviewed    : {s['reviewed']}")
    if s["observed_accuracy"] is not None:
        print(f"  observed accuracy : {s['observed_accuracy']:.1%} "
              f"(on reviewed rows only - not a test-set number)")


if __name__ == "__main__":
    main()
