"""
Feature Store — Real-time and batch feature serving for ML models.

Architecture mirrors Uber Michelangelo, LinkedIn Feathr, and Meta's internal store:
  - Offline store: batch features for training (Parquet / BigQuery)
  - Online store: low-latency feature serving for inference (Redis / DynamoDB)
  - Feature registry: versioned definitions with lineage tracking
  - Point-in-time joins: prevents training/serving skew

Why feature stores matter at FAANG:
  Without a feature store, teams recompute the same features in 10 different places.
  This causes training/serving skew — the #1 silent killer of ML model quality.
  Meta's FBLearner, Google's Feast, and Uber's Michelangelo exist to solve this.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    EMBEDDING = "embedding"
    BOOLEAN = "boolean"


@dataclass
class FeatureDefinition:
    name: str
    feature_type: FeatureType
    description: str
    transform: Optional[Callable] = None   # transformation function
    default_value: Any = None
    tags: list[str] = field(default_factory=list)
    version: str = "1.0"
    owner: str = ""
    ttl_seconds: Optional[int] = None      # None = no expiry


@dataclass
class FeatureVector:
    entity_id: str
    features: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    feature_version: str = "1.0"

    def get(self, feature_name: str, default: Any = None) -> Any:
        return self.features.get(feature_name, default)


class FeatureRegistry:
    """
    Central registry for feature definitions with lineage tracking.
    All features must be registered before they can be stored or retrieved.
    """

    def __init__(self):
        self._features: dict[str, FeatureDefinition] = {}

    def register(self, feature: FeatureDefinition) -> None:
        self._features[feature.name] = feature
        logger.info(f"[FeatureRegistry] Registered: {feature.name} (v{feature.version})")

    def get(self, name: str) -> Optional[FeatureDefinition]:
        return self._features.get(name)

    def list_features(self, tag: Optional[str] = None) -> list[FeatureDefinition]:
        features = list(self._features.values())
        if tag:
            features = [f for f in features if tag in f.tags]
        return features

    def validate_feature_vector(self, vector: FeatureVector) -> list[str]:
        """Returns list of validation errors."""
        errors = []
        for name, value in vector.features.items():
            defn = self._features.get(name)
            if not defn:
                errors.append(f"Unknown feature: {name}")
                continue
            if value is None and defn.default_value is None:
                errors.append(f"Feature {name} is None with no default")
        return errors


class OnlineFeatureStore:
    """
    Low-latency online feature store for real-time inference.
    Target: <5ms p99 latency for feature retrieval.

    Production backend: Redis Cluster or DynamoDB
    This implementation: in-memory dict (swap backend without API changes)
    """

    def __init__(self, registry: FeatureRegistry, default_ttl: int = 3600):
        self._store: dict[str, dict] = {}  # entity_id → {feature_name: (value, expiry)}
        self._registry = registry
        self._default_ttl = default_ttl
        self._hit_count = 0
        self._miss_count = 0

    def write(self, entity_id: str, features: dict[str, Any]) -> None:
        now = time.time()
        if entity_id not in self._store:
            self._store[entity_id] = {}

        for name, value in features.items():
            defn = self._registry.get(name)
            ttl = defn.ttl_seconds if defn and defn.ttl_seconds else self._default_ttl
            expiry = now + ttl if ttl else None

            # Apply transform if defined
            if defn and defn.transform:
                try:
                    value = defn.transform(value)
                except Exception as e:
                    logger.warning(f"[OnlineStore] Transform failed for {name}: {e}")

            self._store[entity_id][name] = (value, expiry)

        logger.debug(f"[OnlineStore] Written {len(features)} features for {entity_id}")

    def read(self, entity_id: str, feature_names: list[str]) -> FeatureVector:
        now = time.time()
        features = {}

        entity_data = self._store.get(entity_id, {})

        for name in feature_names:
            if name in entity_data:
                value, expiry = entity_data[name]
                if expiry is None or now < expiry:
                    features[name] = value
                    self._hit_count += 1
                else:
                    # Expired — use default
                    defn = self._registry.get(name)
                    features[name] = defn.default_value if defn else None
                    self._miss_count += 1
            else:
                # Missing — use default
                defn = self._registry.get(name)
                features[name] = defn.default_value if defn else None
                self._miss_count += 1

        return FeatureVector(entity_id=entity_id, features=features)

    def read_batch(self, entity_ids: list[str], feature_names: list[str]) -> list[FeatureVector]:
        return [self.read(eid, feature_names) for eid in entity_ids]

    def delete(self, entity_id: str) -> None:
        self._store.pop(entity_id, None)

    @property
    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "total_entities": len(self._store),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(self.hit_rate, 4),
        }


class OfflineFeatureStore:
    """
    Batch feature store for training data generation.
    Supports point-in-time joins to prevent training/serving skew.

    Production: Hive / BigQuery / Parquet on S3
    This implementation: in-memory with timestamp-aware lookups
    """

    def __init__(self, registry: FeatureRegistry):
        self._store: list[FeatureVector] = []
        self._registry = registry

    def write_batch(self, vectors: list[FeatureVector]) -> None:
        self._store.extend(vectors)
        logger.info(f"[OfflineStore] Written {len(vectors)} feature vectors | total={len(self._store)}")

    def point_in_time_join(
        self,
        entity_ids: list[str],
        feature_names: list[str],
        as_of_timestamp: float,
    ) -> list[FeatureVector]:
        """
        Critical for preventing training/serving skew.
        Returns the feature values that were available at as_of_timestamp,
        not the latest values (which would leak future information).
        """
        results = []
        for entity_id in entity_ids:
            # Get all records for this entity before the cutoff
            entity_records = [
                v for v in self._store
                if v.entity_id == entity_id and v.timestamp <= as_of_timestamp
            ]

            if not entity_records:
                # Use defaults
                defns = {n: self._registry.get(n) for n in feature_names}
                features = {n: d.default_value if d else None for n, d in defns.items()}
                results.append(FeatureVector(
                    entity_id=entity_id,
                    features=features,
                    timestamp=as_of_timestamp,
                ))
                continue

            # Take most recent record before cutoff
            latest = max(entity_records, key=lambda v: v.timestamp)
            features = {n: latest.features.get(n) for n in feature_names}
            results.append(FeatureVector(
                entity_id=entity_id,
                features=features,
                timestamp=latest.timestamp,
            ))

        logger.info(
            f"[OfflineStore] Point-in-time join: {len(entity_ids)} entities | "
            f"as_of={datetime.fromtimestamp(as_of_timestamp).isoformat()}"
        )
        return results

    def generate_training_dataset(
        self,
        feature_names: list[str],
        label_column: str,
        start_time: float,
        end_time: float,
    ) -> list[dict]:
        """Generate training dataset with features + labels for a time range."""
        records = [
            v for v in self._store
            if start_time <= v.timestamp <= end_time and label_column in v.features
        ]

        dataset = []
        for record in records:
            row = {n: record.features.get(n) for n in feature_names}
            row[label_column] = record.features[label_column]
            row["entity_id"] = record.entity_id
            row["timestamp"] = record.timestamp
            dataset.append(row)

        logger.info(f"[OfflineStore] Generated dataset with {len(dataset)} examples")
        return dataset

    def stats(self) -> dict:
        entity_ids = {v.entity_id for v in self._store}
        return {
            "total_records": len(self._store),
            "unique_entities": len(entity_ids),
            "time_range": {
                "start": min((v.timestamp for v in self._store), default=0),
                "end": max((v.timestamp for v in self._store), default=0),
            } if self._store else {},
        }


class FeatureStore:
    """
    Unified feature store combining online and offline stores.
    Single interface for all feature operations.
    """

    def __init__(self):
        self.registry = FeatureRegistry()
        self.online = OnlineFeatureStore(self.registry)
        self.offline = OfflineFeatureStore(self.registry)

    def register_feature(self, feature: FeatureDefinition) -> None:
        self.registry.register(feature)

    def materialize(self, entity_id: str, features: dict[str, Any]) -> None:
        """Write to both online and offline stores simultaneously."""
        self.online.write(entity_id, features)
        vector = FeatureVector(entity_id=entity_id, features=features)
        self.offline.write_batch([vector])

    def get_online_features(self, entity_id: str, feature_names: list[str]) -> FeatureVector:
        return self.online.read(entity_id, feature_names)

    def get_training_data(
        self,
        feature_names: list[str],
        label_column: str,
        start_time: float,
        end_time: float,
    ) -> list[dict]:
        return self.offline.generate_training_dataset(
            feature_names=feature_names,
            label_column=label_column,
            start_time=start_time,
            end_time=end_time,
        )
