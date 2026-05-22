# Rapport analytique - Maintenance predictive industrielle

## Tache principale
Classification binaire: prediction de `failure_within_24h`, c'est-a-dire identifier les machines susceptibles de tomber en panne dans les 24 prochaines heures.

## Separation train/test
La separation est chronologique: les 80% premieres observations servent a l'entrainement et les 20% dernieres au test. Cette approche limite la fuite d'information temporelle.

## Comparaison quantitative
| accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | average_precision | model | cv_f1_mean | cv_roc_auc_mean | cv_average_precision_mean | type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9933 | 0.9762 | 0.9379 | 0.9567 | 0.9472 | 0.9989 | 0.9858 | xgboost | 0.9017 | 0.9958 | 0.9804 | machine_learning |
| 0.9898 | 0.9339 | 0.9631 | 0.8700 | 0.9142 | 0.9988 | 0.9838 | random_forest | 0.9298 | 0.9961 | 0.9817 | machine_learning |
| 0.9881 | 0.9766 | 0.8627 | 0.9633 | 0.9102 | 0.9956 | 0.9587 | logistic_regression | 0.8433 | 0.9854 | 0.9417 | machine_learning |
| 0.9865 | 0.9772 | 0.8406 | 0.9667 | 0.8992 | 0.9975 | 0.9656 | tensorflow_mlp |  |  |  | deep_learning |
| 0.9859 | 0.8913 | 0.9874 | 0.7833 | 0.8736 | 0.9981 | 0.9780 | gradient_boosting | 0.8955 | 0.9928 | 0.9683 | machine_learning |

## Modele retenu
Le meilleur modele selon le F1-score de test est `xgboost`.

## Variables importantes
| feature | importance_mean | importance_std |
| --- | --- | --- |
| temperature_motor_roll_mean_6 | 0.6470 | 0.0116 |
| rpm_roll_mean_6 | 0.4834 | 0.0087 |
| pressure_level_roll_mean_6 | 0.1052 | 0.0057 |
| vibration_rms_roll_mean_6 | 0.1051 | 0.0077 |
| rpm | 0.0710 | 0.0104 |
| current_phase_avg_roll_mean_6 | 0.0293 | 0.0044 |
| hours_since_maintenance | 0.0172 | 0.0019 |
| vibration_rms | 0.0128 | 0.0046 |

## Interpretation metier
Le rappel mesure la capacite a detecter les pannes a venir; il est critique pour reduire les arrets non planifies. La precision indique la qualite des alertes et aide a limiter les interventions inutiles. Le F1-score sert de compromis pour choisir un modele operationnel.