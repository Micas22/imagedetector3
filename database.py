"""
database.py — SQLite persistence layer.

Replaces two CSV files:
  - .crawler_classification_cache.csv  → classification_cache table
  - runs/<id>/results.csv              → run_results table (optional CSV export remains available)

Schema
------
classification_cache:
  Primary key is (image_hash, classifier_version, threshold, fast_mode, turbo_mode, flag_uncertain)
  so that different run configurations never pollute each other's cache entries.

run_results:
  One row per image per run.  image_hash is stored so the webapp can issue
  "mark as normal" corrections without re-fetching the image bytes.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from constants import CLASSIFIER_VERSION, DB_FILENAME, DEFAULT_TABLE_SCORE_THRESHOLD, ImageResult

# ---------------------------------------------------------------------------
# Connection / init
# ---------------------------------------------------------------------------

_DB_PATH = Path(DB_FILENAME)
_init_lock = threading.Lock()
_initialised = False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # allows concurrent readers
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_init() -> None:
    global _initialised
    if _initialised:
        return
    with _init_lock:
        if _initialised:
            return
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS classification_cache (
                    image_hash          TEXT    NOT NULL,
                    label               TEXT    NOT NULL,
                    score               REAL    NOT NULL,
                    reason              TEXT    NOT NULL,
                    classifier_version  TEXT    NOT NULL,
                    threshold           TEXT    NOT NULL,
                    fast_mode           INTEGER NOT NULL,
                    turbo_mode          INTEGER NOT NULL,
                    flag_uncertain      INTEGER NOT NULL,
                    PRIMARY KEY (
                        image_hash, classifier_version, threshold,
                        fast_mode, turbo_mode, flag_uncertain
                    )
                );

                CREATE TABLE IF NOT EXISTS run_results (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT    NOT NULL,
                    page_url    TEXT    NOT NULL,
                    image_url   TEXT    NOT NULL,
                    image_hash  TEXT    NOT NULL DEFAULT '',
                    label       TEXT    NOT NULL,
                    score       REAL    NOT NULL,
                    reason      TEXT    NOT NULL,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_run_results_run_id
                    ON run_results (run_id);

                CREATE INDEX IF NOT EXISTS idx_cache_hash
                    ON classification_cache (image_hash);
                """
            )
            conn.commit()
        finally:
            conn.close()
        _initialised = True


# ---------------------------------------------------------------------------
# Classification cache
# ---------------------------------------------------------------------------

def load_classification_cache(
    fast_mode: bool,
    turbo_mode: bool = False,
    table_score_threshold: float = DEFAULT_TABLE_SCORE_THRESHOLD,
    flag_uncertain: bool = False,
) -> Dict[str, Tuple[str, float, str]]:
    """Return {image_hash: (label, score, reason)} for matching config."""
    _ensure_init()
    cache: Dict[str, Tuple[str, float, str]] = {}
    try:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT image_hash, label, score, reason
                FROM   classification_cache
                WHERE  classifier_version = ?
                  AND  threshold          = ?
                  AND  fast_mode          = ?
                  AND  turbo_mode         = ?
                  AND  flag_uncertain     = ?
                """,
                (
                    CLASSIFIER_VERSION,
                    f"{table_score_threshold:.4f}",
                    1 if fast_mode else 0,
                    1 if turbo_mode else 0,
                    1 if flag_uncertain else 0,
                ),
            ).fetchall()
            for row in rows:
                cache[row["image_hash"]] = (
                    row["label"],
                    float(row["score"]),
                    row["reason"],
                )
        finally:
            conn.close()
    except Exception:
        pass
    return cache


def save_classification_cache(
    cache: Dict[str, Tuple[str, float, str]],
    fast_mode: bool,
    turbo_mode: bool = False,
    table_score_threshold: float = DEFAULT_TABLE_SCORE_THRESHOLD,
    flag_uncertain: bool = False,
) -> None:
    """Upsert all entries in *cache* into the database."""
    _ensure_init()
    if not cache:
        return
    rows = [
        (
            image_hash,
            label,
            round(score, 6),
            reason,
            CLASSIFIER_VERSION,
            f"{table_score_threshold:.4f}",
            1 if fast_mode else 0,
            1 if turbo_mode else 0,
            1 if flag_uncertain else 0,
        )
        for image_hash, (label, score, reason) in cache.items()
    ]
    try:
        conn = _connect()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO classification_cache
                    (image_hash, label, score, reason,
                     classifier_version, threshold, fast_mode, turbo_mode, flag_uncertain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Run results
# ---------------------------------------------------------------------------

def write_results_db(run_id: str, rows: List[ImageResult]) -> None:
    """Replace all stored rows for *run_id* with *rows* (canonical run snapshot)."""
    _ensure_init()
    if not run_id:
        return
    created_at = datetime.now().isoformat(timespec="seconds")
    data = [
        (
            run_id,
            r.page_url,
            r.image_url,
            r.image_hash,
            r.label,
            round(r.score, 6),
            r.reason,
            created_at,
        )
        for r in rows
    ]
    try:
        conn = _connect()
        try:
            conn.execute("DELETE FROM run_results WHERE run_id = ?", (run_id,))
            if data:
                conn.executemany(
                    """
                    INSERT INTO run_results
                        (run_id, page_url, image_url, image_hash, label, score, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    data,
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def update_run_result_label(
    run_id: str,
    image_url: str,
    label: str,
    score: float,
    reason: str,
) -> None:
    """Update label/score/reason for one row (e.g. manual correction in the webapp)."""
    if not run_id or not image_url:
        return
    _ensure_init()
    try:
        conn = _connect()
        try:
            conn.execute(
                """
                UPDATE run_results
                SET label = ?, score = ?, reason = ?
                WHERE run_id = ? AND image_url = ?
                """,
                (label, round(score, 6), reason, run_id, image_url),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def load_results_db(run_id: str) -> List[Dict]:
    """Return all result rows for *run_id* as plain dicts."""
    _ensure_init()
    try:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT page_url, image_url, image_hash, label, score, reason
                FROM   run_results
                WHERE  run_id = ?
                ORDER  BY id
                """,
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception:
        return []


def list_run_ids() -> List[str]:
    """Return distinct run_ids ordered newest-first."""
    _ensure_init()
    try:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM run_results ORDER BY run_id DESC"
            ).fetchall()
            return [r["run_id"] for r in rows]
        finally:
            conn.close()
    except Exception:
        return []


def get_run_summary(run_id: str) -> Dict:
    """Return a summary dict for a single run (counts + created_at)."""
    _ensure_init()
    try:
        conn = _connect()
        try:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                       AS total,
                    SUM(CASE WHEN label = 'table'    THEN 1 END)  AS tables,
                    SUM(CASE WHEN label = 'normal'   THEN 1 END)  AS normal,
                    SUM(CASE WHEN label = 'uncertain' THEN 1 END) AS uncertain,
                    MIN(created_at)                                AS created_at
                FROM run_results
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row:
                return {
                    "total":     row["total"]     or 0,
                    "tables":    row["tables"]    or 0,
                    "normal":    row["normal"]    or 0,
                    "uncertain": row["uncertain"] or 0,
                    "created_at": row["created_at"] or "",
                }
        finally:
            conn.close()
    except Exception:
        pass
    return {"total": 0, "tables": 0, "normal": 0, "uncertain": 0, "created_at": ""}


def delete_run(run_id: str) -> bool:
    """Delete all rows for *run_id* from run_results. Returns True on success."""
    if not run_id:
        return False
    _ensure_init()
    try:
        conn = _connect()
        try:
            conn.execute("DELETE FROM run_results WHERE run_id = ?", (run_id,))
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception:
        return False


def clear_classification_cache() -> int:
    """Delete every row from classification_cache. Returns number of rows deleted."""
    _ensure_init()
    try:
        conn = _connect()
        try:
            cur = conn.execute("DELETE FROM classification_cache")
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()
    except Exception:
        return 0


def get_cache_stats() -> Dict:
    """Return basic stats about the classification cache."""
    _ensure_init()
    try:
        conn = _connect()
        try:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_entries,
                    SUM(CASE WHEN label = 'table'    THEN 1 END) AS tables,
                    SUM(CASE WHEN label = 'normal'   THEN 1 END) AS normal,
                    SUM(CASE WHEN label = 'uncertain' THEN 1 END) AS uncertain
                FROM classification_cache
                """
            ).fetchone()
            if row:
                return {
                    "total_entries": row["total_entries"] or 0,
                    "tables":        row["tables"]        or 0,
                    "normal":        row["normal"]        or 0,
                    "uncertain":     row["uncertain"]     or 0,
                }
        finally:
            conn.close()
    except Exception:
        pass
    return {"total_entries": 0, "tables": 0, "normal": 0, "uncertain": 0}