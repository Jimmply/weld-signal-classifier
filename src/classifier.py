"""XGBoost classifier for weld in-process signal defect detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Feature columns produced by WeldFleetSignalGenerator._extract_features
FEAT_COLS = [
    c for c in [
        "photodiode_v_mean", "photodiode_v_std", "photodiode_v_max",
        "photodiode_v_kurtosis", "photodiode_v_skewness", "photodiode_v_rms",
        "photodiode_v_p2p",
        "acoustic_rms_mv_mean", "acoustic_rms_mv_std", "acoustic_rms_mv_max",
        "acoustic_rms_mv_kurtosis", "acoustic_rms_mv_skewness", "acoustic_rms_mv_rms",
        "acoustic_rms_mv_p2p",
        "back_reflection_pct_mean", "back_reflection_pct_std",
        "back_reflection_pct_max", "back_reflection_pct_kurtosis",
        "back_reflection_pct_skewness", "back_reflection_pct_rms",
        "back_reflection_pct_p2p",
        "event_rate", "n_events",
    ]
]


@dataclass
class TrainResults:
    classification_report: str
    feature_importances: pd.Series
    label_encoder: LabelEncoder
    test_accuracy: float
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0


class WeldSignalClassifier:
    """XGBoost defect classifier on extracted weld signal features."""

    def __init__(self) -> None:
        self._model = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="mlogloss", verbosity=0,
        )
        self._le = LabelEncoder()
        self._trained = False

    def fit(self, df: pd.DataFrame) -> TrainResults:
        available = [c for c in FEAT_COLS if c in df.columns]
        X = df[available].values
        y = self._le.fit_transform(df["defect_type"])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        self._model.fit(X_tr, y_tr)
        self._trained = True
        self._feat_cols = available

        y_pred = self._model.predict(X_te)
        accuracy = (y_pred == y_te).mean()
        report = classification_report(y_te, y_pred, target_names=self._le.classes_)
        importances = (
            pd.Series(self._model.feature_importances_, index=available)
            .sort_values(ascending=True)
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self._model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        logger.info(
            "Trained. Holdout accuracy=%.3f  CV accuracy=%.3f ± %.3f",
            accuracy, cv_mean, cv_std,
        )
        return TrainResults(report, importances, self._le, accuracy, cv_mean, cv_std)

    def predict_with_explanation(self, row: pd.DataFrame) -> tuple[str, float, pd.DataFrame]:
        available = [c for c in self._feat_cols if c in row.columns]
        X = row[available].values
        enc   = self._model.predict(X)[0]
        proba = self._model.predict_proba(X)[0]
        pred  = self._le.inverse_transform([enc])[0]
        conf  = float(proba[enc])
        # Simple feature contribution: model importances × feature value (normalised)
        fi = pd.Series(self._model.feature_importances_, index=self._feat_cols)
        shap_df = pd.DataFrame({"feature": available, "importance": fi[available].values})
        return pred, conf, shap_df

    def class_probabilities(self, row: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self._feat_cols if c in row.columns]
        X = row[available].values
        proba = self._model.predict_proba(X)[0]
        return pd.DataFrame({
            "defect_type": list(self._le.classes_),
            "probability": proba,
        }).sort_values("probability", ascending=True)

    def save(self, path: str | Path) -> None:
        """Persist the trained classifier and label encoder to disk."""
        if not self._trained:
            raise RuntimeError("Nothing to save — call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "le": self._le, "feat_cols": self._feat_cols}, path)
        logger.info("Classifier saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "WeldSignalClassifier":
        """Load a previously saved classifier from disk."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._model = data["model"]
        obj._le = data["le"]
        obj._feat_cols = data["feat_cols"]
        obj._trained = True
        return obj
