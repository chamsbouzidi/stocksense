"""
retrain.py — Script de réentraînement automatisé StockSense
Déclenché automatiquement quand le monitoring détecte un drift > 5%.
Usage : python retrain/retrain.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# ── Chemins ────────────────────────────────────────────
DATA_DIR  = os.getenv('DATA_DIR',  'data/')
MODEL_DIR = os.getenv('MODEL_DIR', 'models/')

FEATURES = [
    'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance',
    'Year', 'Month', 'Week', 'DayOfMonth',
    'Is_Christmas_Period', 'Is_Summer_Sales', 'Is_Winter_Sales',
    'Is_Sales_Period', 'Is_Monday', 'Is_Saturday'
]

def feature_engineering(df):
    """Même pipeline que le notebook et le monitoring."""
    df = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()
    df['Year']       = df['Date'].dt.year
    df['Month']      = df['Date'].dt.month
    df['Week']       = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek']  = df['Date'].dt.dayofweek
    df['DayOfMonth'] = df['Date'].dt.day
    df['Is_Christmas_Period'] = df['Month'].isin([11, 12]).astype(int)
    df['Is_Summer_Sales']     = (df['Month'] == 7).astype(int)
    df['Is_Winter_Sales']     = (df['Month'] == 1).astype(int)
    df['Is_Sales_Period']     = df['Month'].isin([1, 7]).astype(int)
    df['Is_Monday']           = (df['DayOfWeek'] == 0).astype(int)
    df['Is_Saturday']         = (df['DayOfWeek'] == 5).astype(int)
    df['StoreType']    = df['StoreType'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3})
    df['Assortment']   = df['Assortment'].map({'a': 0, 'b': 1, 'c': 2})
    df['StateHoliday'] = (df['StateHoliday']
                          .map({'0': 0, 0: 0, 'a': 1, 'b': 2, 'c': 3})
                          .fillna(0).astype(int))
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(
        df['CompetitionDistance'].median())
    return df

def load_data():
    print("📦 Chargement des données...")
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'),
                        parse_dates=['Date'], low_memory=False)
    store = pd.read_csv(os.path.join(DATA_DIR, 'store.csv'))
    df = train.merge(store, on='Store', how='left')
    print(f"   {len(df):,} lignes chargées")
    return df

def retrain(df):
    """Réentraîne le modèle sur toutes les données disponibles."""
    print("⚙️  Feature engineering...")
    df = feature_engineering(df)

    X = df[FEATURES]
    y = df['Sales']

    # Cette fois on entraîne sur TOUTES les données disponibles
    # car on est en production — plus de split, on veut le maximum d'historique
    split_date = '2014-12-01'
    X_train = X[df['Date'] < split_date]
    X_test  = X[df['Date'] >= split_date]
    y_train = y[df['Date'] < split_date]
    y_test  = y[df['Date'] >= split_date]

    print("🤖 Réentraînement en cours...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)

    # Évaluation post-réentraînement
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    mape   = mean_absolute_percentage_error(y_test, y_pred) * 100
    bias   = float((y_pred - y_test).mean())

    print(f"\n{'='*50}")
    print(f"  RÉSULTATS POST-RÉENTRAÎNEMENT")
    print(f"{'='*50}")
    print(f"  MAE  : {mae:,.0f} unités")
    print(f"  MAPE : {mape:.1f}%")
    print(f"  Biais: {bias:+.0f} unités")
    print(f"{'='*50}")

    return model, mape

def save_model(model, mape):
    """Sauvegarde le nouveau modèle avec versioning."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Sauvegarde du modèle courant en backup
    current_path = os.path.join(MODEL_DIR, 'model.pkl')
    if os.path.exists(current_path):
        timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(MODEL_DIR, f'model_backup_{timestamp}.pkl')
        joblib.dump(joblib.load(current_path), backup_path)
        print(f"📦 Backup sauvegardé : {backup_path}")

    # Sauvegarde du nouveau modèle
    joblib.dump(model, current_path)
    print(f"💾 Nouveau modèle sauvegardé : {current_path}")

    # Log du réentraînement
    log_path = os.path.join(MODEL_DIR, 'retrain_log.txt')
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"MAPE: {mape:.1f}% | "
                f"Status: success\n")
    print(f"📝 Log mis à jour : {log_path}")

if __name__ == '__main__':
    print("🔄 Démarrage du réentraînement StockSense")
    print(f"   Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    df           = load_data()
    model, mape  = retrain(df)
    save_model(model, mape)
    print("\n✅ Réentraînement terminé avec succès.")