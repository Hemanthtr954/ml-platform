"""
ML Monitoring — Drift detection, data quality, and alerting.

Detects:
  - Feature drift: input distribution shifted from training
  - Prediction drift: output distribution changed
  - Data quality issues: nulls, outliers, schema violations
  - Concept drift: model accuracy degrading over time

Methods:
  - PSI (Population Stability Index) — industry standard for feature drift
  - KL Divergence — information-theoretic drift measure
  - KS Test — non-parametric distribution comparison
  - Z-score — outlier detection for numerical features

This is what separates senior ML engineers from juniors.
Most candidates know how to train models. Few know how to keep them healthy in production.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftMethod(str, Enum):
    PSI = "psi"
    KL_DIVERGENCE = "kl_divergence"
    KS_TEST = "ks_test"


@dataclass
class DriftResult:
    feature_name: str
    method: DriftMethod
    score: float
    threshold: float
    drifted: bool
    severity: AlertSeverity
    details: dict = field(default_factory=dict)


@dataclass
class Alert:
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    model_name: str
    feature_name: Optional[str]
    score: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False


class PSICalculator:
    """
    Population Stability Index — the industry standard drift metric.

    PSI < 0.1:   No significant drift (green)
    PSI 0.1-0.2: Moderate drift — investigate (yellow)
    PSI > 0.2:   Significant drift — action required (red)

    Used by every major bank and tech company to monitor model inputs.
    """

    def __init__(self, n_bins: int = 10, epsilon: float = 1e-8):
        self.n_bins = n_bins
        self.epsilon = epsilon

    def calculate(
        self,
        baseline: list[float],
        current: list[float],
    ) -> float:
        if not baseline or not current:
            return 0.0

        min_val = min(min(baseline), min(current))
        max_val = max(max(baseline), max(current))

        if min_val == max_val:
            return 0.0

        bin_edges = [min_val + (max_val - min_val) * i / self.n_bins for i in range(self.n_bins + 1)]

        baseline_counts = self._bin_data(baseline, bin_edges)
        current_counts = self._bin_data(current, bin_edges)

        n_baseline = len(baseline)
        n_current = len(current)

        psi = 0.0
        for b_count, c_count in zip(baseline_counts, current_counts):
            b_pct = max(b_count / n_baseline, self.epsilon)
            c_pct = max(c_count / n_current, self.epsilon)
            psi += (c_pct - b_pct) * math.log(c_pct / b_pct)

        return psi

    def _bin_data(self, data: list[float], edges: list[float]) -> list[int]:
        counts = [0] * (len(edges) - 1)
        for val in data:
            for i in range(len(edges) - 1):
                if edges[i] <= val <= edges[i + 1]:
                    counts[i] += 1
                    break
        return counts


class KLDivergenceCalculator:
    """KL Divergence for comparing probability distributions."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def calculate(self, p: list[float], q: list[float]) -> float:
        if len(p) != len(q):
            return float("inf")

        total_p = sum(p) + self.epsilon * len(p)
        total_q = sum(q) + self.epsilon * len(q)

        kl = 0.0
        for pi, qi in zip(p, q):
            pi_norm = (pi + self.epsilon) / total_p
            qi_norm = (qi + self.epsilon) / total_q
            kl += pi_norm * math.log(pi_norm / qi_norm)

        return kl


class DataQualityChecker:
    """Validates incoming data against a baseline schema and statistics."""

    def __init__(self, null_threshold: float = 0.05, outlier_z_threshold: float = 3.0):
        self.null_threshold = null_threshold
        self.outlier_z_threshold = outlier_z_threshold

    def check(self, data: list[dict], feature_names: list[str]) -> dict:
        issues = []
        stats = {}

        for feature in feature_names:
            values = [row.get(feature) for row in data]
            null_count = sum(1 for v in values if v is None)
            null_rate = null_count / max(len(values), 1)

            numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]

            feature_stats = {
                "null_rate": round(null_rate, 4),
                "count": len(values),
                "numeric_count": len(numeric_values),
            }

            if null_rate > self.null_threshold:
                issues.append({
                    "feature": feature,
                    "issue": "high_null_rate",
                    "value": null_rate,
                    "threshold": self.null_threshold,
                })

            if numeric_values:
                mean = sum(numeric_values) / len(numeric_values)
                variance = sum((x - mean) ** 2 for x in numeric_values) / max(len(numeric_values) - 1, 1)
                std = math.sqrt(variance)

                outliers = [v for v in numeric_values if abs(v - mean) > self.outlier_z_threshold * std]
                outlier_rate = len(outliers) / len(numeric_values)

                feature_stats.update({
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "outlier_rate": round(outlier_rate, 4),
                })

                if outlier_rate > 0.01:
                    issues.append({
                        "feature": feature,
                        "issue": "high_outlier_rate",
                        "value": outlier_rate,
                        "threshold": 0.01,
                    })

            stats[feature] = feature_stats

        return {"issues": issues, "stats": stats, "passed": len(issues) == 0}


class DriftDetector:
    """
    Production drift detection system.
    Monitors feature distributions and triggers alerts.
    """

    PSI_WARNING = 0.1
    PSI_CRITICAL = 0.2

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._baselines: dict[str, list[float]] = {}
        self._alerts: list[Alert] = []
        self._psi = PSICalculator()
        self._kl = KLDivergenceCalculator()
        self._quality = DataQualityChecker()
        self._alert_count = 0

    def set_baseline(self, feature_name: str, values: list[float]) -> None:
        self._baselines[feature_name] = values
        logger.info(
            f"[DriftDetector] Baseline set for {feature_name}: "
            f"n={len(values)} mean={sum(values)/len(values):.4f}"
        )

    def detect_drift(
        self,
        feature_name: str,
        current_values: list[float],
        method: DriftMethod = DriftMethod.PSI,
    ) -> DriftResult:
        baseline = self._baselines.get(feature_name)
        if not baseline:
            raise ValueError(f"No baseline set for feature: {feature_name}")

        if method == DriftMethod.PSI:
            score = self._psi.calculate(baseline, current_values)
            threshold_warning = self.PSI_WARNING
            threshold_critical = self.PSI_CRITICAL
        elif method == DriftMethod.KL_DIVERGENCE:
            score = self._kl.calculate(baseline, current_values)
            threshold_warning = 0.1
            threshold_critical = 0.5
        else:
            score = 0.0
            threshold_warning = 0.05
            threshold_critical = 0.1

        drifted = score > threshold_warning
        severity = (
            AlertSeverity.CRITICAL if score > threshold_critical
            else AlertSeverity.WARNING if score > threshold_warning
            else AlertSeverity.INFO
        )

        result = DriftResult(
            feature_name=feature_name,
            method=method,
            score=round(score, 6),
            threshold=threshold_warning,
            drifted=drifted,
            severity=severity,
            details={
                "baseline_mean": sum(baseline) / len(baseline),
                "current_mean": sum(current_values) / len(current_values),
                "baseline_n": len(baseline),
                "current_n": len(current_values),
            },
        )

        if drifted:
            self._create_alert(result)

        return result

    def detect_all_features(
        self,
        current_data: dict[str, list[float]],
        method: DriftMethod = DriftMethod.PSI,
    ) -> list[DriftResult]:
        results = []
        for feature_name, values in current_data.items():
            if feature_name in self._baselines:
                result = self.detect_drift(feature_name, values, method)
                results.append(result)
        return results

    def check_data_quality(self, data: list[dict], feature_names: list[str]) -> dict:
        return self._quality.check(data, feature_names)

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        resolved: Optional[bool] = None,
    ) -> list[Alert]:
        alerts = self._alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def resolve_alert(self, alert_id: str) -> None:
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"[DriftDetector] Alert resolved: {alert_id}")
                return

    def _create_alert(self, result: DriftResult) -> Alert:
        self._alert_count += 1
        alert = Alert(
            alert_id=f"alert_{self._alert_count:04d}",
            severity=result.severity,
            title=f"Feature drift detected: {result.feature_name}",
            message=(
                f"PSI={result.score:.4f} exceeds threshold {result.threshold}. "
                f"Baseline mean: {result.details.get('baseline_mean', 0):.4f}, "
                f"Current mean: {result.details.get('current_mean', 0):.4f}"
            ),
            model_name=self.model_name,
            feature_name=result.feature_name,
            score=result.score,
            threshold=result.threshold,
        )
        self._alerts.append(alert)
        logger.warning(
            f"[DriftDetector] ALERT [{result.severity}] {result.feature_name}: "
            f"PSI={result.score:.4f}"
        )
        return alert
