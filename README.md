# 🏠 Prédiction de Prix Immobilier

Projet de régression supervisée pour estimer le prix de vente (`SalePrice`) d'un logement.

## 📊 Dataset
- Source : House Prices - Advanced Regression Techniques
- Fichier : `data/raw/train.csv` (1460 lignes, 81 colonnes)
- Variable cible : `SalePrice`

## 🗂️ Structure
- `data/` : données brutes et transformées
- `notebooks/` : exploration, EDA, feature engineering, modèles
- `src/` : scripts Python réutilisables
- `models/` : modèles entraînés (.pkl / .joblib)
- `dashboard/` : application Streamlit
- `tests/` : tests unitaires

## 🚀 Statut
- [x] Étape 1.1 — Dataset récupéré
- [x] Étape 1.2 — Structure Git
- [x] Étape 1 — Exploration & préparation
- [x] Étape 2 — EDA
- [x] Étape 3 — Feature Engineering
- [x] Étape 4 — Entraînement des modèles
- [x] Étape 5 — Cross-validation 5-fold
- [x] Étape 6 — Évaluation & comparaison
- [ ] Étape 7 — Interprétation avancée
- [ ] Étape 8 — Streamlit
- [ ] Étape 9 — Docker

## Reproduction

Les notebooks doivent être exécutées dans cet ordre :

1. `01_exploration.ipynb` — crée `train.csv`, `train_no_missing.csv` et `train_clean.csv`.
2. `02_eda.ipynb` — produit les analyses exploratoires.
3. `03_feature_engineering.ipynb` — crée les 9 variables dérivées et les artefacts finaux à 209 features.
4. `04_modeling.ipynb` — compare les trois modèles sur validation, puis évalue le meilleur une seule fois sur le test.

Le split final est de 70 % entraînement (`1020` lignes), 15 % validation (`219` lignes) et 15 % test (`219` lignes). Le test reste isolé jusqu'à l'évaluation finale.

Les artefacts générés (`data/processed/*` et les modèles binaires) sont ignorés par Git. Il faut donc régénérer les données et les modèles après un clone propre.

Validation rapide :

```powershell
python -m unittest tests.test_pipeline_integrity -v
```

Le scaler a été entraîné avec une version légèrement différente de scikit-learn dans l'environnement actuel. Pour supprimer l'avertissement de chargement, régénérer les artefacts avec la version de scikit-learn installée dans l'environnement cible.