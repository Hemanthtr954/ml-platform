"""
ML Platform Test Suite — covers all 5 components.
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock

from experiment_tracker.tracker import ExperimentTracker, RunStatus
from model_registry.registry import ModelRegistry, ModelStage, ModelFramework, ModelMetrics
from feature_store.store import FeatureStore, FeatureDefinition, FeatureType, FeatureVector
from monitoring.drift_detector import DriftDetector, DriftMethod, AlertSeverity, PSICalculator
from evaluation.safety_eval import SafetyEvaluator, BehaviorClassifier, SafetyCategory, SafetyTestCase


# ── Experiment Tracker Tests ──────────────────────────────────────────────────

class TestExperimentTracker:
    def setup_method(self, tmp_path=None):
        import tempfile, os
        self.tmp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(storage_dir=self.tmp_dir)

    def test_create_and_complete_run(self):
        run = self.tracker.create_run("test_exp", params={"lr": 0.001})
        assert run.run_id
        assert run.status == RunStatus.CREATED

        self.tracker.start_run(run.run_id)
        self.tracker.log_metric(run.run_id, "loss", 0.5, step=1)
        self.tracker.log_metric(run.run_id, "loss", 0.3, step=2)
        self.tracker.complete_run(run.run_id)

        retrieved = self.tracker.get_run(run.run_id)
        assert retrieved.status == RunStatus.COMPLETED
        assert len(retrieved.metrics["loss"]) == 2

    def test_log_multiple_metrics(self):
        run = self.tracker.create_run("test_exp")
        self.tracker.log_metrics(run.run_id, {"loss": 0.5, "accuracy": 0.85}, step=10)
        retrieved = self.tracker.get_run(run.run_id)
        assert "loss" in retrieved.metrics
        assert "accuracy" in retrieved.metrics

    def test_get_best_metric(self):
        run = self.tracker.create_run("test_exp")
        for i, loss in enumerate([0.9, 0.7, 0.4, 0.5]):
            self.tracker.log_metric(run.run_id, "loss", loss, step=i)
        retrieved = self.tracker.get_run(run.run_id)
        assert retrieved.get_best_metric("loss", mode="min") == 0.4

    def test_fail_run(self):
        run = self.tracker.create_run("test_exp")
        self.tracker.fail_run(run.run_id, "OOM error")
        retrieved = self.tracker.get_run(run.run_id)
        assert retrieved.status == RunStatus.FAILED
        assert "OOM" in retrieved.tags.get("error", "")

    def test_compare_runs(self):
        run1 = self.tracker.create_run("exp")
        run2 = self.tracker.create_run("exp")
        self.tracker.log_metric(run1.run_id, "loss", 0.5)
        self.tracker.log_metric(run2.run_id, "loss", 0.3)
        comparison = self.tracker.compare_runs([run1.run_id, run2.run_id], ["loss"])
        assert len(comparison) == 2

    def test_list_runs_filter_by_experiment(self):
        self.tracker.create_run("exp_a")
        self.tracker.create_run("exp_b")
        runs_a = self.tracker.list_runs(experiment_name="exp_a")
        assert len(runs_a) == 1


# ── Model Registry Tests ──────────────────────────────────────────────────────

class TestModelRegistry:
    def setup_method(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(storage_dir=self.tmp_dir)

    def test_register_and_retrieve(self):
        v = self.registry.register_model(
            model_name="llm_v1",
            version="1.0",
            artifact_path="/checkpoints/v1",
            framework=ModelFramework.HUGGINGFACE,
            metrics=ModelMetrics(eval_loss=0.45),
        )
        assert v.version == "1.0"
        assert v.stage == ModelStage.DEVELOPMENT

    def test_promotion_workflow(self):
        self.registry.register_model("model", "1.0", "/path", ModelFramework.PYTORCH)
        self.registry.promote("model", "1.0", ModelStage.STAGING, "ci_bot", "passed tests")
        self.registry.promote("model", "1.0", ModelStage.PRODUCTION, "engineer", "approved")

        prod = self.registry.get_production_model("model")
        assert prod is not None
        assert prod.version == "1.0"
        assert prod.stage == ModelStage.PRODUCTION

    def test_only_one_production_model(self):
        self.registry.register_model("model", "1.0", "/path1", ModelFramework.PYTORCH)
        self.registry.register_model("model", "2.0", "/path2", ModelFramework.PYTORCH)

        self.registry.promote("model", "1.0", ModelStage.STAGING)
        self.registry.promote("model", "1.0", ModelStage.PRODUCTION)
        self.registry.promote("model", "2.0", ModelStage.STAGING)
        self.registry.promote("model", "2.0", ModelStage.PRODUCTION)

        production_models = [
            v for v in self.registry.list_versions("model")
            if v.stage == ModelStage.PRODUCTION
        ]
        assert len(production_models) == 1
        assert production_models[0].version == "2.0"

    def test_invalid_promotion_raises(self):
        self.registry.register_model("model", "1.0", "/path", ModelFramework.PYTORCH)
        with pytest.raises(ValueError):
            self.registry.promote("model", "1.0", ModelStage.PRODUCTION)  # Skip staging

    def test_traffic_split(self):
        self.registry.register_model("model", "1.0", "/p1", ModelFramework.PYTORCH)
        self.registry.register_model("model", "2.0", "/p2", ModelFramework.PYTORCH)
        self.registry.set_traffic_split("model", {"1.0": 0.9, "2.0": 0.1})
        # Should not raise
        assert "1.0" in self.registry._traffic_splits["model"]

    def test_promotion_history(self):
        self.registry.register_model("model", "1.0", "/path", ModelFramework.PYTORCH)
        self.registry.promote("model", "1.0", ModelStage.STAGING, "bot", "CI passed")
        history = self.registry.promotion_history("model")
        assert len(history) == 1
        assert history[0]["by"] == "bot"


# ── Feature Store Tests ───────────────────────────────────────────────────────

class TestFeatureStore:
    def setup_method(self):
        self.store = FeatureStore()
        self.store.register_feature(FeatureDefinition(
            name="age", feature_type=FeatureType.FLOAT, description="User age", default_value=0.0
        ))
        self.store.register_feature(FeatureDefinition(
            name="score", feature_type=FeatureType.FLOAT, description="User score", default_value=0.0
        ))

    def test_materialize_and_retrieve(self):
        self.store.materialize("user_123", {"age": 28.0, "score": 0.85})
        vector = self.store.get_online_features("user_123", ["age", "score"])
        assert vector.features["age"] == 28.0
        assert vector.features["score"] == 0.85

    def test_missing_entity_returns_defaults(self):
        vector = self.store.get_online_features("unknown_user", ["age"])
        assert vector.features["age"] == 0.0

    def test_point_in_time_join(self):
        t1 = time.time() - 100
        t2 = time.time() - 50
        now = time.time()

        self.store.offline.write_batch([
            FeatureVector("u1", {"age": 25.0}, timestamp=t1),
            FeatureVector("u1", {"age": 26.0}, timestamp=t2),
        ])

        result = self.store.offline.point_in_time_join(["u1"], ["age"], as_of_timestamp=t1 + 10)
        assert result[0].features["age"] == 25.0  # Should return t1 value, not t2

    def test_online_store_hit_rate(self):
        self.store.materialize("u1", {"age": 30.0})
        self.store.get_online_features("u1", ["age"])   # hit
        self.store.get_online_features("u2", ["age"])   # miss
        assert self.store.online.hit_rate > 0


# ── Drift Detector Tests ──────────────────────────────────────────────────────

class TestDriftDetector:
    def setup_method(self):
        self.detector = DriftDetector("test_model")
        self.baseline = [float(i) for i in range(100)]
        self.detector.set_baseline("feature_x", self.baseline)

    def test_no_drift_on_same_distribution(self):
        result = self.detector.detect_drift("feature_x", self.baseline, DriftMethod.PSI)
        assert not result.drifted
        assert result.score < 0.1

    def test_drift_detected_on_shifted_distribution(self):
        shifted = [float(i + 50) for i in range(100)]
        result = self.detector.detect_drift("feature_x", shifted, DriftMethod.PSI)
        assert result.drifted
        assert result.score > 0.1

    def test_critical_drift_creates_alert(self):
        very_different = [float(i * 10) for i in range(100)]
        self.detector.detect_drift("feature_x", very_different, DriftMethod.PSI)
        alerts = self.detector.get_alerts()
        assert len(alerts) > 0

    def test_psi_zero_for_identical(self):
        psi = PSICalculator()
        data = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        score = psi.calculate(data, data)
        assert score < 0.01

    def test_data_quality_check(self):
        data = [{"feature_x": None}, {"feature_x": 1.0}, {"feature_x": 2.0}]
        result = self.detector.check_data_quality(data, ["feature_x"])
        assert "issues" in result
        assert not result["passed"]  # null rate > threshold


# ── Safety Evaluator Tests ────────────────────────────────────────────────────

class TestBehaviorClassifier:
    def setup_method(self):
        self.classifier = BehaviorClassifier()

    def test_detects_refusal(self):
        response = "I can't help with that request."
        behavior, score = self.classifier.classify(response, "refuse")
        assert behavior == "refused"
        assert score == 1.0

    def test_detects_compliance(self):
        response = "Sure! Here is the information you requested: ..."
        behavior, score = self.classifier.classify(response, "comply")
        assert behavior == "complied"
        assert score == 1.0

    def test_refusal_when_expected_compliance_scores_zero(self):
        response = "I cannot help with that."
        behavior, score = self.classifier.classify(response, "comply")
        assert score == 0.0

    def test_compliance_when_expected_refusal_scores_zero(self):
        response = "Here is how to do it: step 1, step 2, step 3..."
        behavior, score = self.classifier.classify(response, "refuse")
        assert score == 0.0
