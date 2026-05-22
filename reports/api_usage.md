# API Maintenance Predictive

## Objectif

L'API permet d'envoyer une observation machine et de recevoir une probabilite de panne dans les 24 prochaines heures.

Le modele utilise est **XGBoost**, selectionne comme meilleur modele selon le F1-score.

## Lancement

Depuis la racine du projet:

```powershell
.\venv\Scripts\uvicorn.exe api.main:app --reload
```

Documentation interactive:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### GET /health

Verifie que l'API fonctionne et que le modele est disponible.

Exemple de reponse:

```json
{
  "status": "ok",
  "model_available": true
}
```

### GET /model-info

Retourne les informations principales sur le modele charge.

Exemple de reponse:

```json
{
  "target": "failure_within_24h",
  "production_model": "xgboost",
  "best_overall_by_f1": "xgboost",
  "feature_count": 37,
  "decision_threshold": 0.5,
  "model_available": true
}
```

### POST /predict

Effectue une prediction pour une observation machine.

Exemple d'URL:

```text
http://127.0.0.1:8000/predict?threshold=0.5
```

Exemple de corps JSON:

```json
{
  "timestamp": "2024-01-01 12:00:00",
  "machine_id": 1,
  "machine_type": "CNC",
  "vibration_rms": 0.82,
  "temperature_motor": 49.5,
  "current_phase_avg": 5.1,
  "pressure_level": 23.6,
  "rpm": 860.9,
  "operating_mode": "idle",
  "hours_since_maintenance": 274.0,
  "ambient_temp": 13.9
}
```

Exemple de reponse:

```json
{
  "failure_probability": 0.037145163863897324,
  "failure_within_24h_prediction": 0,
  "threshold": 0.5,
  "model": "xgboost"
}
```

Interpretation:

- `failure_probability`: probabilite estimee de panne dans les 24h.
- `failure_within_24h_prediction`: prediction finale selon le seuil choisi.
- `threshold`: seuil de decision utilise.
- `model`: modele utilise pour la prediction.

## Remarque metier

Le seuil peut etre ajuste selon la strategie de maintenance:

- seuil plus bas: plus d'alertes, moins de pannes manquees;
- seuil plus haut: moins d'alertes, mais risque plus eleve de manquer une panne.

Dans ce projet, le seuil par defaut est fixe a `0.5`.
