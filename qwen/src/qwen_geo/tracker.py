from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    run_id: int
    run_name: str
    status: str
    started_at: str
    finished_at: str | None
    parent_run_id: int | None
    score: float | None
    promoted: bool
    description: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    artifact_dir: str | None
    error_message: str | None


class ExperimentTracker:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(db_path))
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                parent_run_id INTEGER,
                score REAL,
                promoted INTEGER NOT NULL DEFAULT 0,
                description TEXT NOT NULL,
                config_json TEXT NOT NULL,
                metrics_json TEXT,
                artifact_dir TEXT,
                error_message TEXT
            )
            """
        )
        self.connection.commit()

    def start_run(
        self,
        *,
        run_name: str,
        started_at: str,
        description: str,
        config: dict[str, Any],
        parent_run_id: int | None = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO runs (
                run_name, status, started_at, parent_run_id, description, config_json
            ) VALUES (?, 'running', ?, ?, ?, ?)
            """,
            (run_name, started_at, parent_run_id, description, json.dumps(config, sort_keys=True)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        *,
        status: str,
        finished_at: str,
        score: float | None,
        promoted: bool,
        metrics: dict[str, Any] | None,
        artifact_dir: str | None,
        error_message: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE runs
            SET status = ?, finished_at = ?, score = ?, promoted = ?, metrics_json = ?, artifact_dir = ?, error_message = ?
            WHERE run_id = ?
            """,
            (
                status,
                finished_at,
                score,
                1 if promoted else 0,
                json.dumps(metrics, sort_keys=True) if metrics is not None else None,
                artifact_dir,
                error_message,
                run_id,
            ),
        )
        self.connection.commit()

    def list_runs(self) -> list[RunRecord]:
        rows = self.connection.execute("SELECT * FROM runs ORDER BY run_id ASC").fetchall()
        return [self._row_to_record(row) for row in rows]

    def completed_runs(self) -> list[RunRecord]:
        rows = self.connection.execute(
            """
            SELECT * FROM runs
            WHERE status IN ('completed', 'promoted')
            ORDER BY (score IS NULL) ASC, score DESC, run_id DESC
            """
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def best_run(self) -> RunRecord | None:
        row = self.connection.execute(
            """
            SELECT * FROM runs
            WHERE status IN ('completed', 'promoted') AND score IS NOT NULL
            ORDER BY score DESC, run_id DESC
            LIMIT 1
            """
        ).fetchone()
        return self._row_to_record(row) if row else None

    def champion_score(self) -> float | None:
        best = self.best_run()
        return best.score if best else None

    def recent_failures(self, limit: int = 5) -> list[RunRecord]:
        rows = self.connection.execute(
            "SELECT * FROM runs WHERE status = 'failed' ORDER BY run_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=int(row["run_id"]),
            run_name=row["run_name"],
            status=row["status"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            parent_run_id=row["parent_run_id"],
            score=row["score"],
            promoted=bool(row["promoted"]),
            description=row["description"],
            config=json.loads(row["config_json"]),
            metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else {},
            artifact_dir=row["artifact_dir"],
            error_message=row["error_message"],
        )

    def close(self) -> None:
        self.connection.close()
