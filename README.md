# 🛍️ StockSense — Demand Forecasting for Retail

Solution de prévision de la demande développée dans le cadre d'une thèse 
RNCP Data Engineer. Prédit les ventes journalières par point de vente 
à partir d'un historique de 2 ans sur 1 115 magasins.

## Résultats
| Métrique | Valeur |
|---|---|
| MAE | 1 325 unités |
| MAPE | 20,3% |
| Biais | -99 unités (quasi nul) |

## Structure
| Dossier | Contenu |
|---|---|
| `/notebooks` | Notebook d'exploration et d'entraînement |
| `/src` | Script d'entraînement standalone |
| `/tests` | Tests unitaires du modèle |
| `/models` | Modèle sérialisé (.pkl) |
| `/api` | API FastAPI de serving *(étape suivante)* |
| `/monitoring` | Rapports Evidently *(étape suivante)* |
| `/retrain` | Script de réentraînement *(étape suivante)* |

## Installation
```bash
pip install -r requirements.txt
```

## Entraînement
```bash
python src/train.py
```

## Tests
```bash
pytest tests/
```

## Stack technique
Python · XGBoost · FastAPI · Docker · GitHub Actions · Evidently
