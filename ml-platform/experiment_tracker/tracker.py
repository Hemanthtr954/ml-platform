"""
Experiment Tracker — MLflow-style experiment tracking built from scratch.

Tracks:
  - Hyperparameters (learning rate, batch size, model config)
  - Metrics per step/epoch (loss, accuracy, perplexity)
  - Artifacts (model checkpoints, eval results, plots)
  - System metrics (GPU utilization, memory, throughput)
  - Git commit hash for full reproducibility

Why build this instead of using MLflow?
  FAANG companies build internal versions because:
  - Full control over storage backend (S3, GCS, internal blob)
  - Custom metric aggregation and alerting
  - Integration with internal model registry and serving
  - No external dependencies in air-gapped environments
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class MetricPoint:
    key: str
    value: float
    step: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExperimentRun:
    run_id: str
    experiment_name: str
    status: RunStatus
    params: dict = field(default_factory=dict)
    metrics: dict[str, list[MetricPoint]] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    git_commit: Optional[str] = None
    notes: str = ""

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def get_best_metric(self, key: str, mode: str = "min") -> Optional[float]:
        if key not in self.metrics or not self.metrics[key]:
            return None
        values = [m.value for m in self.metrics[key]]
        return min(values) if mode == "min" else max(values)

    def get_latest_metric(self, key: str) -> Optional[float]:
        if key not in self.metrics or not self.metrics[key]:
            return None
        return self.metrics[key][-1].value


class ExperimentTracker:
    """
    Production experiment tracker with SQLite backend.
    Supports concurrent runs, metric streaming, and artifact management.

    In production (Google/Meta scale):
      - Backend: Spanner / BigQuery / internal blob store
      - Streaming metrics: Kafka → real-time dashboard
      - Artifacts: GCS / S3 with lifecycle policies
    """

    def __init__(self, storage_dir: str = "./experiments"):
        self._storage = Path(storage_dir)
        self._storage.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, ExperimentRun] = {}
        self._load_existing_runs()

    def create_run(
        self,
        experiment_name: str,
        params: dict = None,
        tags: dict = None,
        notes: str = "",
    ) -> ExperimentRun:
        run_id = str(uuid.uuid4())[:8]
        git_commit = self._get_git_commit()

        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            status=RunStatus.CREATED,
            params=params or {},
            tags=tags or {},
            git_commit=git_commit,
            notes=notes,
        )
        self._runs[run_id] = run
        self._save_run(run)

        logger.info(f"[Tracker] Created run {run_id} for experiment '{experiment_name}'")
        return run

    def start_run(self, run_id: str) -> None:
        run = self._get_run(run_id)
        run.status = RunStatus.RUNNING
        run.start_time = time.time()
        self._save_run(run)

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        run = self._get_run(run_id)
        run.params[key] = value
        self._save_run(run)

    def log_params(self, run_id: str, params: dict) -> None:
        run = self._get_run(run_id)
        run.params.update(params)
        self._save_run(run)

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        run = self._get_run(run_id)
        if key not in run.metrics:
            run.metrics[key] = []
        run.metrics[key].append(MetricPoint(key=key, value=value, step=step))
        self._save_run(run)

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int = 0) -> None:
        for key, value in metrics.items():
            self.log_metric(run_id, key, value, step)

    def log_artifact(self, run_id: str, artifact_path: str) -> None:
        run = self._get_run(run_id)
        run.artifacts.append(artifact_path)
        self._save_run(run)

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        run = self._get_run(run_id)
        run.tags[key] = value
        self._save_run(run)

    def complete_run(self, run_id: str) -> None:
        run = self._get_run(run_id)
        run.status = RunStatus.COMPLETED
        run.end_time = time.time()
        self._save_run(run)
        logger.info(
            f"[Tracker] Run {run_id} completed in "
            f"{run.duration_seconds:.1f}s"
        )

    def fail_run(self, run_id: str, error: str = "") -> None:
        run = self._get_run(run_id)
        run.status = RunStatus.FAILED
        run.end_time = time.time()
        run.tags["error"] = error[:500]
        self._save_run(run)
        logger.error(f"[Tracker] Run {run_id} failed: {error[:100]}")

    def get_run(self, run_id: str) -> ExperimentRun:
        return self._get_run(run_id)

    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        status: Optional[RunStatus] = None,
    ) -> list[ExperimentRun]:
        runs = list(self._runs.values())
        if experiment_name:
            runs = [r for r in runs if r.experiment_name == experiment_name]
        if status:
            runs = [r for r in runs if r.status == status]
        return sorted(runs, key=lambda r: r.start_time, reverse=True)

    def compare_runs(
        self,
        run_ids: list[str],
        metrics: list[str],
    ) -> list[dict]:
        results = []
        for run_id in run_ids:
            run = self._get_run(run_id)
            row = {
                "run_id": run_id,
                "experiment": run.experiment_name,
                "status": run.status,
                "params": run.params,
            }
            for metric in metrics:
                row[f"{metric}_best"] = run.get_best_metric(metric)
                row[f"{metric}_latest"] = run.get_latest_metric(metric)
            results.append(row)
        return results

    def get_best_run(
        self,
        experiment_name: str,
        metric: str,
        mode: str = "min",
    ) -> Optional[ExperimentRun]:
        runs = self.list_runs(
            experiment_name=experiment_name,
            status=RunStatus.COMPLETED,
        )
        if not runs:
            return None

        def score(run):
            val = run.get_best_metric(metric, mode)
            return val if val is not None else float("inf") if mode == "min" else float("-inf")

        return min(runs, key=score) if mode == "min" else max(runs, key=score)

    def _get_run(self, run_id: str) -> ExperimentRun:
        if run_id not in self._runs:
            raise KeyError(f"Run '{run_id}' not found")
        return self._runs[run_id]

    def _save_run(self, run: ExperimentRun) -> None:
        run_dir = self._storage / run.experiment_name / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Serialize metrics
        serializable = {
            "run_id": run.run_id,
            "experiment_name": run.experiment_name,
            "status": run.status.value,
            "params": run.params,
            "metrics": {
                k: [{"key": m.key, "value": m.value, "step": m.step, "timestamp": m.timestamp}
                    for m in v]
                for k, v in run.metrics.items()
            },
            "tags": run.tags,
            "artifacts": run.artifacts,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "git_commit": run.git_commit,
            "notes": run.notes,
        }

        with open(run_dir / "run.json", "w") as f:
            json.dump(serializable, f, indent=2)

    def _load_existing_runs(self) -> None:
        for run_file in self._storage.rglob("run.json"):
            try:
                with open(run_file) as f:
                    data = json.load(f)
                metrics = {
                    k: [MetricPoint(**m) for m in v]
                    for k, v in data.get("metrics", {}).items()
                }
                run = ExperimentRun(
                    run_id=data["run_id"],
                    experiment_name=data["experiment_name"],
                    status=RunStatus(data["status"]),
                    params=data.get("params", {}),
                    metrics=metrics,
                    tags=data.get("tags", {}),
                    artifacts=data.get("artifacts", []),
                    start_time=data.get("start_time", 0),
                    end_time=data.get("end_time"),
                    git_commit=data.get("git_commit"),
                    notes=data.get("notes", ""),
                )
                self._runs[run.run_id] = run
            except Exception as e:
                logger.warning(f"[Tracker] Failed to load run from {run_file}: {e}")

    def _get_git_commit(self) -> Optional[str]:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=2
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
