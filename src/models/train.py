import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:  # pragma: no cover
    tf = None
    keras = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.feature_engineering import build_features, get_model_columns
from src.data.load_data import load_raw_data
from src.data.preprocess import make_preprocessor, split_temporal
from src.models.evaluate import (
    classification_metrics,
    save_diagnostic_plots,
    save_metrics,
    save_report,
)
from src.models.save_model import save_artifact


ARTIFACT_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TARGET = "failure_within_24h"
RANDOM_STATE = 42


def build_sklearn_models(pos_weight: float) -> dict[str, Pipeline]:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.07,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
        )
    return models


def evaluate_sklearn_models(X_train, y_train, X_test, y_test):
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    rows = []
    fitted_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, estimator in build_sklearn_models(pos_weight).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["f1", "roc_auc", "average_precision"],
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = classification_metrics(y_test, y_pred, y_proba)
        metrics.update(
            {
                "model": name,
                "cv_f1_mean": cv_scores["test_f1"].mean(),
                "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
                "cv_average_precision_mean": cv_scores["test_average_precision"].mean(),
                "type": "machine_learning",
            }
        )
        rows.append(metrics)
        fitted_models[name] = pipeline
    return rows, fitted_models


def evaluate_deep_learning(X_train, y_train, X_test, y_test):
    if keras is None:
        return None

    tf.random.set_seed(RANDOM_STATE)
    preprocessor = make_preprocessor(X_train)
    X_train_arr = preprocessor.fit_transform(X_train).astype("float32")
    X_test_arr = preprocessor.transform(X_test).astype("float32")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train_arr.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.Precision(), keras.metrics.Recall()],
    )
    neg, pos = np.bincount(y_train)
    class_weight = {0: 1.0, 1: float(neg / max(pos, 1))}
    model.fit(
        X_train_arr,
        y_train,
        validation_split=0.15,
        epochs=25,
        batch_size=256,
        class_weight=class_weight,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
    )
    y_proba = model.predict(X_test_arr, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = classification_metrics(y_test, y_pred, y_proba)
    metrics.update(
        {
            "model": "tensorflow_mlp",
            "cv_f1_mean": np.nan,
            "cv_roc_auc_mean": np.nan,
            "cv_average_precision_mean": np.nan,
            "type": "deep_learning",
        }
    )

    save_artifact(preprocessor, ARTIFACT_DIR / "deep_learning_preprocessor.joblib")
    model.save(ARTIFACT_DIR / "deep_learning_model.keras")
    return metrics


def save_feature_importance(best_model: Pipeline, X_test, y_test) -> pd.DataFrame:
    result = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="f1",
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    output_path = REPORT_DIR / "feature_importance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    return importance


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for col in display_df.select_dtypes(include=["float"]).columns:
        display_df[col] = display_df[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = list(display_df.columns)
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in display_df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(rows)


def write_modeling_summary(metrics_df: pd.DataFrame, best_name: str, feature_importance: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    top_features = feature_importance.head(8)
    lines = [
        "# Rapport analytique - Maintenance predictive industrielle",
        "",
        "## Tache principale",
        "Classification binaire: prediction de `failure_within_24h`, c'est-a-dire identifier les machines susceptibles de tomber en panne dans les 24 prochaines heures.",
        "",
        "## Separation train/test",
        "La separation est chronologique: les 80% premieres observations servent a l'entrainement et les 20% dernieres au test. Cette approche limite la fuite d'information temporelle.",
        "",
        "## Comparaison quantitative",
        dataframe_to_markdown(metrics_df),
        "",
        "## Modele retenu",
        f"Le meilleur modele selon le F1-score de test est `{best_name}`.",
        "",
        "## Variables importantes",
        dataframe_to_markdown(top_features),
        "",
        "## Interpretation metier",
        "Le rappel mesure la capacite a detecter les pannes a venir; il est critique pour reduire les arrets non planifies. La precision indique la qualite des alertes et aide a limiter les interventions inutiles. Le F1-score sert de compromis pour choisir un modele operationnel.",
    ]
    (REPORT_DIR / "rapport_analytique.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_raw_data()
    features = build_features(df)
    X_train_all, X_test_all, y_train, y_test = split_temporal(features, target=TARGET)

    model_cols = get_model_columns(features)
    X_train = X_train_all[model_cols]
    X_test = X_test_all[model_cols]

    metrics, fitted_models = evaluate_sklearn_models(X_train, y_train, X_test, y_test)
    dl_metrics = evaluate_deep_learning(X_train, y_train, X_test, y_test)
    if dl_metrics is not None:
        metrics.append(dl_metrics)

    metrics_df = save_metrics(metrics, REPORT_DIR / "model_metrics.csv")
    best_sklearn_row = metrics_df[metrics_df["model"].isin(fitted_models.keys())].iloc[0]
    best_name = best_sklearn_row["model"]
    best_model = fitted_models[best_name]

    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    save_artifact(best_model, ARTIFACT_DIR / "best_failure_classifier.joblib")
    save_report(y_test, y_pred, REPORT_DIR / "classification_report.json")
    save_diagnostic_plots(best_model, X_test, y_test, y_pred, FIGURE_DIR)
    importance = save_feature_importance(best_model, X_test, y_test)
    write_modeling_summary(metrics_df, best_name, importance)

    metadata = {
        "target": TARGET,
        "production_model": best_name,
        "best_overall_by_f1": metrics_df.iloc[0]["model"],
        "feature_columns": model_cols,
        "decision_threshold": 0.5,
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics_df.to_string(index=False))
    print(f"\nProduction model saved: {ARTIFACT_DIR / 'best_failure_classifier.joblib'}")


if __name__ == "__main__":
    main()
