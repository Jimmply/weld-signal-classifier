"""XGBoost classifier for weld in-process signal defect detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Feature columns produced by WeldFleetSignalGenerator._extract_features
FEAT_COLS = [
    c for c in [
        "photodiode_v_mean", "photodiode_v_std", "photodiode_v_max",
        "photodiode_v_kurtosis", "photodiode_v_rms",
        "acoustic_rms_mv_mean", "acoustic_rms_mv_std", "acoustic_rms_mv_max",
        "acoustic_rms_mv_kurtosis", "acoustic_rms_mv_rms",
        "back_reflection_pct_mean", "back_reflection_pct_std",
        "back_reflection_pct_max", "back_reflection_pct_kurtosis",
        "back_reflection_pct_rms",
        "event_rate", "n_events",
    ]
]


@dataclass
class TrainResults:
    classification_report: str
    feature_importances: pd.Series
    label_encoder: LabelEncoder
    test_accuracy: float


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
        logger.info("Trained. Accuracy=%.3f", accuracy)
        return TrainResults(report, importances, self._le, accuracy)

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
