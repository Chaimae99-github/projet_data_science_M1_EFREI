from pathlib import Path

import joblib


def save_artifact(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str | Path):
    return joblib.load(Path(path))
