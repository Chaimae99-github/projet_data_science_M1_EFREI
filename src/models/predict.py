import pandas as pd

from src.data.feature_engineering import build_features, get_model_columns


def predict_failure(model, rows: pd.DataFrame) -> pd.DataFrame:
    features = build_features(rows)
    X = features[get_model_columns(features)]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "failure_probability": probabilities,
            "failure_within_24h_prediction": predictions,
        }
    )
