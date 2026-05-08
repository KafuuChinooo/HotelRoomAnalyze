from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from config import MODEL_FEATURE_COLUMNS, MODELS_DIR, TARGET


MODEL_PATH = MODELS_DIR / "hotel_cancellation_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in MODEL_FEATURE_COLUMNS if col in df.columns]
    return df[available].copy()


def build_model_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    classifier = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _top_model_signals(model: Pipeline, X: pd.DataFrame, y: pd.Series, limit: int = 20) -> list[dict[str, float | str]]:
    sample_size = min(5000, len(X))
    X_sample = X.sample(sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    importances = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="roc_auc",
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )
    rows = pd.DataFrame({"feature": X.columns, "importance": importances.importances_mean}).sort_values(
        "importance", ascending=False
    )
    rows = rows.head(limit)
    return [
        {"feature": str(row.feature), "importance": float(row.importance)}
        for row in rows.itertuples(index=False)
    ]


def train_cancellation_model(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> dict:
    X = _feature_frame(df)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = build_model_pipeline(X_train)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "model_path": str(model_path),
        "rows": int(len(df)),
        "features": list(X.columns),
        "target_positive_rate": round(float(y.mean()), 4),
        "test_rows": int(len(y_test)),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "top_model_signals": _top_model_signals(model, X_test, y_test),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_cancellation(model: Pipeline, row: dict) -> float:
    frame = pd.DataFrame([row])
    return float(model.predict_proba(frame)[0, 1])
