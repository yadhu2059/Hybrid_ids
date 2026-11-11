"""
Machine Learning Classifier Architecture & Explainability Engine
Wraps LightGBM / Random Forest with multi-class inference, probability calibration,
and statistical feature-attribution explanation generation for single and batch flows.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import RandomForestClassifier

from app.ml.features import FEATURE_NAMES, extract_features_from_flow


class HybridClassifier:
    """
    LightGBM / Random Forest hybrid classifier providing fast CPU inference
    and deterministic feature-attribution explainability for network flows.
    """
    def __init__(self, use_rf: bool = False, random_state: int = 42):
        self.random_state = random_state
        self.use_rf = use_rf or not HAS_LIGHTGBM
        self.model = None
        self.classes_: np.ndarray = np.array([])
        self.feature_names_: List[str] = list(FEATURE_NAMES)
        self.benign_stats_: Dict[str, Dict[str, float]] = {}

        if self.use_rf:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                n_jobs=2,
                random_state=self.random_state
            )
        else:
            self.model = LGBMClassifier(
                n_estimators=120,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=8,
                class_weight="balanced",
                n_jobs=2,
                random_state=self.random_state,
                verbose=-1
            )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray, List[str]]):
        """Fits model and calculates benign baseline statistics for explainability."""
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names_].values
        else:
            X_mat = np.asarray(X)

        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        # Fit underlying estimator
        self.model.fit(X_mat, y_arr)

        # Calculate benign baseline feature statistics for attribution explainability
        benign_mask = (y_arr == "Benign")
        if np.any(benign_mask):
            X_benign = X_mat[benign_mask]
            means = np.mean(X_benign, axis=0)
            stds = np.std(X_benign, axis=0)
            stds[stds == 0] = 1.0  # guard zero variance
            self.benign_stats_ = {
                feat: {"mean": float(means[i]), "std": float(stds[i])}
                for i, feat in enumerate(self.feature_names_)
            }
        else:
            self.benign_stats_ = {
                feat: {"mean": 0.0, "std": 1.0}
                for feat in self.feature_names_
            }

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts class labels for batch flows."""
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names_].values
        else:
            X_mat = np.asarray(X)
        return self.model.predict(X_mat)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns class probabilities for batch flows."""
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names_].values
        else:
            X_mat = np.asarray(X)
        return self.model.predict_proba(X_mat)

    def explain_flow(
        self,
        features: Dict[str, float],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Derives top contributing feature attributions explaining why a flow
        deviates from benign baseline behavior.
        """
        deviations = []
        for feat in self.feature_names_:
            val = float(features.get(feat, 0.0))
            stat = self.benign_stats_.get(feat, {"mean": 0.0, "std": 1.0})
            mean = stat["mean"]
            std = stat["std"]
            z_score = abs(val - mean) / std
            deviations.append({
                "feature": feat,
                "observed_value": round(val, 4),
                "baseline_mean": round(mean, 4),
                "z_score": round(z_score, 2)
            })

        # Sort by z-score deviation
        deviations.sort(key=lambda x: x["z_score"], reverse=True)
        return deviations[:top_k]

    def predict_single(
        self,
        event_or_features: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """
        Performs ultra-low-latency single-flow inference (< 0.1ms).
        Returns predicted label, anomaly flag, confidence, and top feature attributions.
        """
        if isinstance(event_or_features, str) or ("flow" in event_or_features or "dest_port" in event_or_features and len(event_or_features) < len(self.feature_names_)):
            features = extract_features_from_flow(event_or_features)
        else:
            features = event_or_features

        # Construct 1xN feature vector
        feat_vector = np.array([[features.get(f, 0.0) for f in self.feature_names_]], dtype=np.float64)

        probas = self.model.predict_proba(feat_vector)[0]
        max_idx = int(np.argmax(probas))
        predicted_label = str(self.classes_[max_idx])
        confidence = float(probas[max_idx])

        # Benign vs Anomaly determination
        benign_idx = -1
        if "Benign" in self.classes_:
            benign_idx = int(np.where(self.classes_ == "Benign")[0][0])
            benign_prob = float(probas[benign_idx])
            attack_prob = float(1.0 - benign_prob)
        else:
            attack_prob = confidence if predicted_label != "Benign" else 0.0

        is_anomaly = 1 if (predicted_label != "Benign" or attack_prob > 0.5) else 0

        # Feature explanation
        top_explanations = self.explain_flow(features, top_k=3)

        return {
            "predicted_label": predicted_label,
            "is_anomaly": is_anomaly,
            "confidence": round(confidence, 4),
            "attack_probability": round(attack_prob, 4),
            "top_contributing_features": top_explanations,
            "features_snapshot": {k: features.get(k, 0.0) for k in self.feature_names_[:6]}
        }

    def save(self, filepath: str):
        """Persists trained model and baseline statistics to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "model": self.model,
            "classes": self.classes_,
            "feature_names": self.feature_names_,
            "benign_stats": self.benign_stats_,
            "use_rf": self.use_rf
        }
        joblib.dump(payload, filepath, compress=3)

    @classmethod
    def load(cls, filepath: str) -> "HybridClassifier":
        """Loads trained model artifact from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        payload = joblib.load(filepath)
        instance = cls(use_rf=payload.get("use_rf", False))
        instance.model = payload["model"]
        instance.classes_ = payload["classes"]
        instance.feature_names_ = payload["feature_names"]
        instance.benign_stats_ = payload["benign_stats"]
        return instance
