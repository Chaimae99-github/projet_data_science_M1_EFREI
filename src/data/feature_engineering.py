import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


SENSOR_COLUMNS = [
    "vibration_rms",
    "temperature_motor",
    "pressure_level",
    "rpm",
    "current_phase_avg",
    "ambient_temp",
    "hours_since_maintenance",
]

TARGET_COLUMNS = [
    "failure_within_24h",
    "failure_type",
    "rul_hours",
    "estimated_repair_cost",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "timestamp" not in result.columns:
        return result

    ts = pd.to_datetime(result["timestamp"], errors="coerce")
    result["hour"] = ts.dt.hour
    result["dayofweek"] = ts.dt.dayofweek
    result["month"] = ts.dt.month
    result["is_weekend"] = result["dayofweek"].isin([5, 6]).astype(int)
    return result


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    sort_cols = ["machine_id", "timestamp"] if "timestamp" in result.columns else ["machine_id"]
    result = result.sort_values(sort_cols)

    group_key = "machine_id" if "machine_id" in result.columns else None
    for col in [c for c in SENSOR_COLUMNS if c in result.columns]:
        if group_key:
            grouped = result.groupby(group_key, group_keys=False)[col]
            result[f"{col}_roll_mean_6"] = grouped.rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
            result[f"{col}_roll_std_6"] = grouped.rolling(6, min_periods=2).std().reset_index(level=0, drop=True)
            result[f"{col}_diff_1"] = grouped.diff()
        else:
            result[f"{col}_roll_mean_6"] = result[col].rolling(6, min_periods=1).mean()
            result[f"{col}_roll_std_6"] = result[col].rolling(6, min_periods=2).std()
            result[f"{col}_diff_1"] = result[col].diff()

    trend_cols = [c for c in result.columns if c.endswith("_diff_1")]
    rolling_std_cols = [c for c in result.columns if c.endswith("_roll_std_6")]
    result[trend_cols + rolling_std_cols] = result[trend_cols + rolling_std_cols].fillna(0)
    result["anomaly_trend"] = result[trend_cols].abs().mean(axis=1) if trend_cols else 0.0
    return result


def add_health_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    required = ["vibration_rms", "temperature_motor", "pressure_level", "rpm", "anomaly_trend"]
    available = [c for c in required if c in result.columns]
    if not available:
        result["health_score"] = np.nan
        return result

    weights = {
        "vibration_rms": 0.30,
        "temperature_motor": 0.25,
        "pressure_level": 0.20,
        "rpm": 0.15,
        "anomaly_trend": 0.10,
    }
    scaler = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(result[available].fillna(result[available].median(numeric_only=True))),
        columns=available,
        index=result.index,
    )
    risk = sum(weights[col] * normalized[col] for col in available)
    result["health_score"] = (1 - risk).clip(0, 1)
    return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    result = add_time_features(df)
    result = add_trend_features(result)
    result = add_health_score(result)
    return result.sort_values("timestamp") if "timestamp" in result.columns else result


def get_model_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(TARGET_COLUMNS + ["timestamp","machine_id"])
    return [col for col in df.columns if col not in excluded]


