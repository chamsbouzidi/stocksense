"""
train.py — Script d'entraînement standalone StockSense
Reproduit le pipeline complet du notebook en script exécutable.
Usage : python src/train.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# ── Chemins ────────────────────────────────────────────
DATA_DIR  = os.getenv('DATA_DIR', 'data/')
MODEL_DIR = os.getenv('MODEL_DIR', 'models/')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    print("📦 Chargement des données...")
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'),
                        parse_dates=['Date'], low_memory=False)
    store = pd.read_csv(os.path.join(DATA_DIR, 'store.csv'))
    df = train.merge(store, on='Store', how='left')
    print(f"   {len(df):,} lignes chargées")
    return df

def feature_engineering(df):
    print("⚙️  Feature engineering...")

    # Filtrage
    df = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()

    # Décomposition temporelle
    df['Year']       = df['Date'].dt.year
    df['Month']      = df['Date'].dt.month
    df['Week']       = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek']  = df['Date'].dt.dayofweek
    df['DayOfMonth'] = df['Date'].dt.day

    # Features métier
    df['Is_Christmas_Period'] = df['Month'].isin([11, 12]).astype(int)
    df['Is_Summer_Sales']     = (df['Month'] == 7).astype(int)
    df['Is_Winter_Sales']     = (df['Month'] == 1).astype(int)
    df['Is_Sales_Period']     = df['Month'].isin([1, 7]).astype(int)
    df['Is_Monday']           = (df['DayOfWeek'] == 0).astype(int)
    df['Is_Saturday']         = (df['DayOfWeek'] == 5).astype(int)

    # Encodage
    df['StoreType']    = df['StoreType'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3})
    df['Assortment']   = df['Assortment'].map({'a': 0, 'b': 1, 'c': 2})
    df['StateHoliday'] = (df['StateHoliday']
                          .map({'0': 0, 0: 0, 'a': 1, 'b': 2, 'c': 3})
                          .fillna(0).astype(int))

    # Imputation
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(
        df['CompetitionDistance'].median())

    print(f"   {len(df):,} lignes après filtrage")
    return df

def train_model(df):
    print("🤖 Entraînement du modèle...")

    features = [
        'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'StoreType', 'Assortment', 'CompetitionDistance',
        'Year', 'Month', 'Week', 'DayOfMonth',
        'Is_Christmas_Period', 'Is_Summer_Sales', 'Is_Winter_Sales',
        'Is_Sales_Period', 'Is_Monday', 'Is_Saturday'
    ]

    X, y = df[features], df['Sales']

    split_date = '2014-12-01'
    X_train = X[df['Date'] < split_date]
    X_test  = X[df['Date'] >= split_date]
    y_train = y[df['Date'] < split_date]
    y_test  = y[df['Date'] >= split_date]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)

    # Évaluation
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    bias = float((y_pred - y_test).mean())

    print(f"   MAE  : {mae:,.0f}")
    print(f"   MAPE : {mape:.1f}%")
    print(f"   Biais: {bias:+.0f}")

    return model

def save_model(model):
    path = os.path.join(MODEL_DIR, 'model.pkl')
    joblib.dump(model, path)
    print(f"💾 Modèle sauvegardé → {path}")

if __name__ == '__main__':
    df    = load_data()
    df    = feature_engineering(df)
    model = train_model(df)
    save_model(model)
    print("✅ Pipeline terminé.")
