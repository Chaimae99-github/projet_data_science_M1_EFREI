# Modelisation et interpretation

## Objectif

L'objectif est de predire `failure_within_24h`, donc de savoir si une machine risque de tomber en panne dans les 24 prochaines heures.

## Resultat principal

Le meilleur modele est **XGBoost** avec:

- F1-score: **0.9505**
- Recall: **0.9600**
- Precision: **0.9412**
- ROC AUC: **0.9991**
- Average Precision: **0.9886**

Ce modele est retenu car il offre le meilleur compromis entre detection des pannes et limitation des fausses alertes.

## Comparaison des modeles

La regression logistique obtient le meilleur recall, avec **0.9667**, mais sa precision est plus faible (**0.8631**). Elle detecte donc legerement plus de pannes, mais genere davantage de fausses alertes.

Random Forest et Gradient Boosting ont une precision tres elevee, mais leur recall est plus faible. Ils sont donc plus prudents: ils alertent moins souvent, mais manquent davantage de pannes.

Le modele Deep Learning TensorFlow MLP satisfait l'exigence du cahier des charges. Ses performances sont bonnes, mais inferieures a XGBoost sur le F1-score.

## Matrice de confusion du modele retenu

Sur le jeu de test, XGBoost donne:

- vrais negatifs: 4491
- faux positifs: 18
- faux negatifs: 12
- vrais positifs: 288

Le modele detecte donc **288 pannes sur 300**, ce qui est tres satisfaisant pour une application de maintenance predictive.

## Importance des variables

Les variables les plus importantes sont principalement des moyennes mobiles:

1. `temperature_motor_roll_mean_6`
2. `rpm_roll_mean_6`
3. `vibration_rms_roll_mean_6`
4. `pressure_level_roll_mean_6`
5. `rpm`

Cela montre que les tendances recentes des capteurs sont plus informatives que certaines mesures instantanees. Cette observation est coherente avec le domaine industriel: une panne est souvent precedee par une degradation progressive du comportement machine.

## Conclusion

XGBoost est le meilleur choix pour la mise en production dans ce projet. Il combine une tres bonne detection des pannes, un faible nombre de fausses alertes et une interpretation possible via l'importance des variables.

Pour la soutenance, il faut insister sur le fait que le recall est prioritaire dans ce contexte, car une panne non detectee peut entrainer des couts importants.
