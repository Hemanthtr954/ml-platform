"""
ML Platform API — Unified FastAPI interface for the full ML lifecycle.

Endpoints:
  Experiment Tracker:
    POST /experiments/runs              — Create experiment run
    POST /experiments/runs/{id}/metrics — Log metrics
    GET  /experiments/runs              — List runs
    GET  /experiments/runs/{id}         — Get run details
    GET  /experiments/compare           — Compare runs

  Model Registry:
    POST /models/register               — Register model version
    POST /models/{name}/{version}/promote — Promote model stage
    GET  /models/{name}/production      — Get production model
    GET  /models/{name}/versions        — List versions
    POST /models/{name}/traffic-split   — Configure A/B split

  Feature Store:
    POST /features/register             — Register feature definition
    POST /features/materialize          — Write features
    GET  /features/{entity_id}          — Get online features

  Monitoring:
    POST /monitoring/baseline           — Set feature baseline
    POST /monitoring/drift              — Run drift detection
    GET  /monitoring/alerts             — Get active alerts

  Evaluation:
    POST /eval/safety                   — Run safety eval suite
    GET  /health                        — Health check
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────

tracker = None
registry = None
feature_store = None
drift_detector = None
safety_evaluator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tracker, registry, feature_store, drift_detector, safety_evaluator

    from experiment_tracker.tracker import ExperimentTracker
    from model_registry.registry import ModelRegistry
    from feature_store.store import FeatureStore
    from monitoring.drift_detector import DriftDetector

    tracker = ExperimentTracker(storage_dir="./experiments")
    registry = ModelRegistry(storage_dir="./model_registry")
    feature_store = FeatureStore()
    drift_detector = DriftDetector(model_name="default")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from openai import OpenAI
        from evaluation.safety_eval import SafetyEvaluator
        llm = OpenAI(api_key=openai_key)
        safety_evaluator = SafetyEvaluator(llm_client=llm)
        logger.info("Safety evaluator initialized")

    logger.info("ML Platform started")
    yield
    logger.info("ML Platform shutting down")


app = FastAPI(
    title="ML Platform",
    description="Production ML lifecycle platform: experiment tracking, model registry, feature store, drift detection, safety evals",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Request Models ─────────────────────────────────────────────────────────────

class CreateRunRequest(BaseModel):
    experiment_name: str
    params: dict = Field(default_factory=dict)
    tags: dict = Field(default_factory=dict)
    notes: str = ""

class LogMetricsRequest(BaseModel):
    metrics: dict[str, float]
    step: int = 0

class RegisterModelRequest(BaseModel):
    model_name: str
    version: str
    artifact_path: str
    framework: str = "pytorch"
    metrics: dict = Field(default_factory=dict)
    params: dict = Field(default_factory=dict)
    run_id: Optional[str] = None
    description: str = ""

class PromoteModelRequest(BaseModel):
    to_stage: str
    promoted_by: str = "api"
    reason: str = ""

class RegisterFeatureRequest(BaseModel):
    name: str
    feature_type: str
    description: str
    default_value: Optional[float] = None
    ttl_seconds: Optional[int] = None
    tags: list[str] = Field(default_factory=list)

class MaterializeRequest(BaseModel):
    entity_id: str
    features: dict[str, float]

class SetBaselineRequest(BaseModel):
    feature_name: str
    values: list[float]

class DetectDriftRequest(BaseModel):
    feature_name: str
    current_values: list[float]
    method: str = "psi"

class SafetyEvalRequest(BaseModel):
    model_name: str
    model_version: str
    system_prompt: str = "You are a helpful AI assistant."


# ── Experiment Tracker Endpoints ───────────────────────────────────────────────

@app.post("/experiments/runs")
async def create_run(request: CreateRunRequest):
    run = tracker.create_run(
        experiment_name=request.experiment_name,
        params=request.params,
        tags=request.tags,
        notes=request.notes,
    )
    tracker.start_run(run.run_id)
    return {"run_id": run.run_id, "experiment_name": run.experiment_name, "status": run.status}

@app.post("/experiments/runs/{run_id}/metrics")
async def log_metrics(run_id: str, request: LogMetricsRequest):
    try:
        tracker.log_metrics(run_id, request.metrics, step=request.step)
        return {"status": "logged", "run_id": run_id, "metrics": request.metrics}
    except KeyError:
        raise HTTPException(404, f"Run '{run_id}' not found")

@app.post("/experiments/runs/{run_id}/complete")
async def complete_run(run_id: str):
    try:
        tracker.complete_run(run_id)
        return {"status": "completed", "run_id": run_id}
    except KeyError:
        raise HTTPException(404, f"Run '{run_id}' not found")

@app.get("/experiments/runs")
async def list_runs(experiment_name: Optional[str] = None):
    runs = tracker.list_runs(experiment_name=experiment_name)
    return {"runs": [
        {
            "run_id": r.run_id,
            "experiment_name": r.experiment_name,
            "status": r.status,
            "params": r.params,
            "duration_seconds": r.duration_seconds,
            "git_commit": r.git_commit,
        }
        for r in runs
    ]}

@app.get("/experiments/runs/{run_id}")
async def get_run(run_id: str):
    try:
        run = tracker.get_run(run_id)
        return {
            "run_id": run.run_id,
            "experiment_name": run.experiment_name,
            "status": run.status,
            "params": run.params,
            "metrics": {
                k: [{"step": m.step, "value": m.value} for m in v]
                for k, v in run.metrics.items()
            },
            "tags": run.tags,
            "duration_seconds": run.duration_seconds,
            "git_commit": run.git_commit,
            "notes": run.notes,
        }
    except KeyError:
        raise HTTPException(404, f"Run '{run_id}' not found")


# ── Model Registry Endpoints ───────────────────────────────────────────────────

@app.post("/models/register")
async def register_model(request: RegisterModelRequest):
    from model_registry.registry import ModelFramework, ModelMetrics
    try:
        framework = ModelFramework(request.framework)
    except ValueError:
        raise HTTPException(400, f"Unknown framework: {request.framework}")

    metrics = ModelMetrics(**{k: v for k, v in request.metrics.items() if hasattr(ModelMetrics, k) or True})
    version = registry.register_model(
        model_name=request.model_name,
        version=request.version,
        artifact_path=request.artifact_path,
        framework=framework,
        params=request.params,
        run_id=request.run_id,
        description=request.description,
    )
    return {"model_name": version.model_name, "version": version.version, "stage": version.stage}

@app.post("/models/{model_name}/{version}/promote")
async def promote_model(model_name: str, version: str, request: PromoteModelRequest):
    from model_registry.registry import ModelStage
    try:
        stage = ModelStage(request.to_stage)
        v = registry.promote(model_name, version, stage, request.promoted_by, request.reason)
        return {"model_name": model_name, "version": version, "stage": v.stage}
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/models/{model_name}/production")
async def get_production(model_name: str):
    v = registry.get_production_model(model_name)
    if not v:
        raise HTTPException(404, f"No production model for '{model_name}'")
    return {"model_name": v.model_name, "version": v.version, "artifact_path": v.artifact_path}

@app.get("/models/{model_name}/versions")
async def list_versions(model_name: str):
    versions = registry.list_versions(model_name)
    return {"versions": [{"version": v.version, "stage": v.stage, "created_at": v.created_at} for v in versions]}

@app.get("/models")
async def list_models():
    return {"models": registry.list_models()}


# ── Feature Store Endpoints ───────────────────────────────────────────────────

@app.post("/features/register")
async def register_feature(request: RegisterFeatureRequest):
    from feature_store.store import FeatureDefinition, FeatureType
    try:
        feature_type = FeatureType(request.feature_type)
    except ValueError:
        raise HTTPException(400, f"Unknown feature type: {request.feature_type}")
    feature = FeatureDefinition(
        name=request.name,
        feature_type=feature_type,
        description=request.description,
        default_value=request.default_value,
        ttl_seconds=request.ttl_seconds,
        tags=request.tags,
    )
    feature_store.register_feature(feature)
    return {"status": "registered", "name": request.name}

@app.post("/features/materialize")
async def materialize_features(request: MaterializeRequest):
    feature_store.materialize(request.entity_id, request.features)
    return {"status": "materialized", "entity_id": request.entity_id, "features": list(request.features.keys())}

@app.get("/features/{entity_id}")
async def get_features(entity_id: str, features: str = ""):
    feature_names = features.split(",") if features else list(feature_store.registry._features.keys())
    vector = feature_store.get_online_features(entity_id, feature_names)
    return {"entity_id": entity_id, "features": vector.features}


# ── Monitoring Endpoints ──────────────────────────────────────────────────────

@app.post("/monitoring/baseline")
async def set_baseline(request: SetBaselineRequest):
    drift_detector.set_baseline(request.feature_name, request.values)
    return {"status": "baseline_set", "feature": request.feature_name, "n": len(request.values)}

@app.post("/monitoring/drift")
async def detect_drift(request: DetectDriftRequest):
    from monitoring.drift_detector import DriftMethod
    try:
        method = DriftMethod(request.method)
        result = drift_detector.detect_drift(request.feature_name, request.current_values, method)
        return {
            "feature": result.feature_name,
            "method": result.method,
            "score": result.score,
            "threshold": result.threshold,
            "drifted": result.drifted,
            "severity": result.severity,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/monitoring/alerts")
async def get_alerts(resolved: Optional[bool] = False):
    alerts = drift_detector.get_alerts(resolved=resolved)
    return {"alerts": [
        {"id": a.alert_id, "severity": a.severity, "title": a.title,
         "feature": a.feature_name, "score": a.score, "timestamp": a.timestamp}
        for a in alerts
    ]}


# ── Evaluation Endpoints ──────────────────────────────────────────────────────

@app.post("/eval/safety")
async def run_safety_eval(request: SafetyEvalRequest):
    if not safety_evaluator:
        raise HTTPException(503, "Safety evaluator not initialized — OPENAI_API_KEY required")
    report = safety_evaluator.evaluate(
        model_name=request.model_name,
        model_version=request.model_version,
        system_prompt=request.system_prompt,
    )
    return {
        "model_name": report.model_name,
        "model_version": report.model_version,
        "total_cases": report.total_cases,
        "passed": report.passed,
        "failed": report.failed,
        "pass_rate": report.pass_rate,
        "overall_score": report.overall_score,
        "is_safe_for_production": report.is_safe_for_production,
        "by_category": report.by_category,
        "critical_failures": len(report.critical_failures),
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "components": {
            "experiment_tracker": tracker is not None,
            "model_registry": registry is not None,
            "feature_store": feature_store is not None,
            "drift_detector": drift_detector is not None,
            "safety_evaluator": safety_evaluator is not None,
        }
    }
