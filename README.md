# Projet Data Science M1 EFREI - Maintenance predictive industrielle

Objectif principal: predire `failure_within_24h`, c'est-a-dire identifier les equipements susceptibles de tomber en panne dans les 24 prochaines heures.

## Livrables couverts

- EDA dans `notebooks/01_eda.ipynb`
- Pipeline de features et preprocessing dans `src/data`
- Comparaison d'au moins 4 modeles dans `src/models/train.py`
- Modele Deep Learning TensorFlow inclus
- Separation train/test chronologique
- Cross-validation sur les modeles scikit-learn
- Metriques, rapport, figures et importance des features dans `reports`
- Dashboard Streamlit dans `dashboard/app.py`
- API FastAPI optionnelle dans `api/main.py`

## Lancer l'entrainement

```powershell
.\venv\Scripts\python.exe src\models\train.py
```

Artefacts generes:

- `models/best_failure_classifier.joblib`
- `models/model_metadata.json`
- `models/deep_learning_model.keras`
- `reports/model_metrics.csv`
- `reports/feature_importance.csv`
- `reports/classification_report.json`
- `reports/figures/*.png`
- `reports/rapport_analytique.md`

## Lancer le dashboard

```powershell
.\venv\Scripts\streamlit.exe run dashboard\app.py
```

## Lancer l'API

```powershell
.\venv\Scripts\uvicorn.exe api.main:app --reload
```

Endpoint principal:

- `GET /health`
- `POST /predict`

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

## Structure conseillee du rapport

1. Contexte metier et objectif
2. Description du dataset
3. EDA: qualite des donnees, distributions, correlations, desequilibre de classe
4. Feature engineering: tendances, variables temporelles, health score
5. Methodologie: split chronologique, preprocessing, metriques
6. Modelisation: modeles compares et Deep Learning
7. Resultats: comparaison quantitative et choix du modele
8. Interpretabilite: importance des variables, lecture metier
9. Dashboard et API
10. Limites et ameliorations possibles
