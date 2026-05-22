import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.feature_engineering import build_features, get_model_columns
from src.data.load_data import load_raw_data
from src.models.save_model import load_artifact


st.set_page_config(page_title="Maintenance predictive", layout="wide")

MODEL_PATH = PROJECT_ROOT / "models" / "best_failure_classifier.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "feature_importance.csv"


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return build_features(load_raw_data())


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return load_artifact(MODEL_PATH)
    return None


@st.cache_data
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


df = load_dataset()
model = load_model()
metadata = load_metadata()

st.title("Maintenance predictive industrielle")
st.caption("Prediction des pannes dans les 24 prochaines heures")

with st.sidebar:
    st.header("Filtres")
    machine_types = sorted(df["machine_type"].dropna().unique()) if "machine_type" in df else []
    selected_types = st.multiselect("Type machine", machine_types, default=machine_types)
    operating_modes = sorted(df["operating_mode"].dropna().unique()) if "operating_mode" in df else []
    selected_modes = st.multiselect("Mode operationnel", operating_modes, default=operating_modes)
    threshold = st.slider("Seuil d'alerte", 0.05, 0.95, 0.50, 0.05)
    max_rows = st.slider("Observations a scorer", 100, 5000, 1000, 100)

filtered = df.copy()
if selected_types:
    filtered = filtered[filtered["machine_type"].isin(selected_types)]
if selected_modes:
    filtered = filtered[filtered["operating_mode"].isin(selected_modes)]

if filtered.empty:
    st.warning("Aucune observation ne correspond aux filtres selectionnes.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Observations", f"{len(filtered):,}".replace(",", " "))
col2.metric("Taux panne 24h", f"{filtered['failure_within_24h'].mean():.1%}")
col3.metric("Health score moyen", f"{filtered['health_score'].mean():.2f}")
col4.metric("Machines", filtered["machine_id"].nunique())

tab_overview, tab_model, tab_prediction = st.tabs(["Surveillance", "Modeles", "Prediction"])

with tab_overview:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Evolution des capteurs")
        sensor = st.selectbox(
            "Capteur",
            ["vibration_rms", "temperature_motor", "pressure_level", "rpm", "health_score"],
        )
        chart_df = filtered.sort_values("timestamp")[["timestamp", sensor]].dropna().set_index("timestamp")
        st.line_chart(chart_df)
    with right:
        st.subheader("Repartition cible")
        st.bar_chart(filtered["failure_within_24h"].value_counts().sort_index())
        st.subheader("Types de panne")
        st.bar_chart(filtered["failure_type"].value_counts().head(10))

    st.subheader("Etat le plus recent par machine")
    latest_machine_state = (
        filtered.sort_values("timestamp")
        .groupby("machine_id", as_index=False)
        .tail(1)[
            [
                "timestamp",
                "machine_id",
                "machine_type",
                "operating_mode",
                "vibration_rms",
                "temperature_motor",
                "rpm",
                "health_score",
                "failure_within_24h",
            ]
        ]
        .sort_values(["failure_within_24h", "health_score"], ascending=[False, True])
    )
    st.dataframe(
        latest_machine_state,
        width="stretch",
        hide_index=True,
        column_config={
            "health_score": st.column_config.ProgressColumn(
                "health_score",
                min_value=0,
                max_value=1,
                format="%.2f",
            ),
            "failure_within_24h": st.column_config.CheckboxColumn("panne_24h"),
        },
    )

with tab_model:
    production_model = metadata.get("production_model", "modele non disponible")
    feature_count = len(metadata.get("feature_columns", []))
    st.subheader("Modele en production")
    model_col1, model_col2, model_col3 = st.columns(3)
    model_col1.metric("Modele retenu", str(production_model))
    model_col2.metric("Nombre de features", feature_count)
    model_col3.metric("Seuil par defaut", metadata.get("decision_threshold", 0.5))

    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        st.subheader("Comparaison quantitative")
        display_metrics = metrics[
            [
                "model",
                "type",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
                "cv_f1_mean",
            ]
        ].sort_values("f1", ascending=False)
        st.dataframe(
            display_metrics,
            width="stretch",
            hide_index=True,
            column_config={
                "accuracy": st.column_config.NumberColumn(format="%.4f"),
                "precision": st.column_config.NumberColumn(format="%.4f"),
                "recall": st.column_config.NumberColumn(format="%.4f"),
                "f1": st.column_config.NumberColumn(format="%.4f"),
                "roc_auc": st.column_config.NumberColumn(format="%.4f"),
                "average_precision": st.column_config.NumberColumn(format="%.4f"),
                "cv_f1_mean": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.bar_chart(metrics.set_index("model")[["f1", "recall", "precision", "roc_auc"]])
    else:
        st.warning("Lance d'abord `python src/models/train.py` pour generer les metriques.")

    if IMPORTANCE_PATH.exists():
        importance = pd.read_csv(IMPORTANCE_PATH).head(15)
        st.subheader("Importance des variables")
        st.bar_chart(importance.set_index("feature")["importance_mean"])

    figure_cols = st.columns(3)
    for column, filename, label in zip(
        figure_cols,
        ["confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png"],
        ["Matrice de confusion", "ROC", "Precision-rappel"],
    ):
        path = PROJECT_ROOT / "reports" / "figures" / filename
        if path.exists():
            column.image(str(path), caption=label, width="stretch")

with tab_prediction:
    if model is None:
        st.warning("Aucun modele trouve. Lance `python src/models/train.py`.")
    else:
        st.subheader("Priorisation des interventions")
        scoring_df = filtered.sort_values("timestamp").tail(int(max_rows)).copy()
        X_score = scoring_df[get_model_columns(scoring_df)]
        scoring_df["failure_probability"] = model.predict_proba(X_score)[:, 1]
        scoring_df["alert"] = scoring_df["failure_probability"] >= threshold

        alert_count = int(scoring_df["alert"].sum())
        alert_rate = alert_count / max(len(scoring_df), 1)
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        pred_col1.metric("Observations scorees", len(scoring_df))
        pred_col2.metric("Alertes au seuil choisi", alert_count)
        pred_col3.metric("Taux d'alerte", f"{alert_rate:.1%}")

        latest_scored = (
            scoring_df.sort_values("timestamp")
            .groupby("machine_id", as_index=False)
            .tail(1)
            .sort_values("failure_probability", ascending=False)
        )
        top_risk = latest_scored[
            [
                "timestamp",
                "machine_id",
                "machine_type",
                "operating_mode",
                "health_score",
                "failure_probability",
                "alert",
            ]
        ].head(20)

        st.subheader("Observations des dernires prédictions")
        st.dataframe(
            top_risk,
            width="stretch",
            hide_index=True,
            column_config={
                "health_score": st.column_config.ProgressColumn(
                    "health_score",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                ),
                "failure_probability": st.column_config.ProgressColumn(
                    "probabilite_panne",
                    min_value=0,
                    max_value=1,
                    format="%.1%",
                ),
                "alert": st.column_config.CheckboxColumn("alerte"),
            },
        )

        st.subheader("Scores par observation")
        st.dataframe(
            scoring_df[
                [
                    "timestamp",
                    "machine_id",
                    "machine_type",
                    "operating_mode",
                    "health_score",
                    "failure_probability",
                    "alert",
                ]
            ].sort_values("failure_probability", ascending=False),
            width="stretch",
            hide_index=True,
            column_config={
                "health_score": st.column_config.ProgressColumn(
                    "health_score",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                ),
                "failure_probability": st.column_config.ProgressColumn(
                    "probabilite_panne",
                    min_value=0,
                    max_value=1,
                    format="%.1%",
                ),
                "alert": st.column_config.CheckboxColumn("alerte"),
            },
        )
