"""
monitor.py — Monitoring de la performance du modèle StockSense
Génère un rapport HTML Evidently qui détecte la dégradation du modèle.
Usage : python monitoring/monitor.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ── Chargement du modèle ───────────────────────────────
MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.pkl')
model = joblib.load(MODEL_PATH)

FEATURES = [
    'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance',
    'Year', 'Month', 'Week', 'DayOfMonth',
    'Is_Christmas_Period', 'Is_Summer_Sales', 'Is_Winter_Sales',
    'Is_Sales_Period', 'Is_Monday', 'Is_Saturday'
]

def feature_engineering(df):
    """Même pipeline que le notebook."""
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

def load_and_prepare():
    """Charge les données et crée deux périodes pour comparaison."""
    train = pd.read_csv('data/train.csv', parse_dates=['Date'], low_memory=False)
    store = pd.read_csv('data/store.csv')
    df = train.merge(store, on='Store', how='left')
    df = feature_engineering(df)

    # Période de référence = données d'entraînement (passé)
    reference = df[df['Date'] < '2014-12-01'][FEATURES + ['Sales']].copy()

    # Période courante = données de test (plus récent)
    current   = df[df['Date'] >= '2014-12-01'][FEATURES + ['Sales']].copy()

    # Ajouter les prédictions
    reference['prediction'] = model.predict(reference[FEATURES])
    current['prediction']   = model.predict(current[FEATURES])

    return reference, current

def generate_report(reference, current):
    """Génère un rapport HTML de monitoring."""
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

    os.makedirs('monitoring/reports', exist_ok=True)
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'monitoring/reports/report_{timestamp}.html'

    # Calcul des métriques
    mape_ref  = mean_absolute_percentage_error(reference['Sales'], reference['prediction']) * 100
    mape_cur  = mean_absolute_percentage_error(current['Sales'],   current['prediction'])   * 100
    mae_ref   = mean_absolute_error(reference['Sales'], reference['prediction'])
    mae_cur   = mean_absolute_error(current['Sales'],   current['prediction'])
    bias_ref  = float((reference['prediction'] - reference['Sales']).mean())
    bias_cur  = float((current['prediction']   - current['Sales']).mean())
    drift     = mape_cur - mape_ref
    status    = "⚠️ ALERTE — Réentraînement recommandé" if drift > 5 else "✅ Modèle stable"
    color     = "#e74c3c" if drift > 5 else "#27ae60"

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>StockSense — Rapport de Monitoring</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f8f9fa; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .status {{ background: {color}; color: white; padding: 15px 20px; border-radius: 8px; font-size: 18px; font-weight: bold; margin: 20px 0; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .card h3 {{ margin: 0 0 15px; color: #7f8c8d; font-size: 13px; text-transform: uppercase; }}
        .metric {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .sub {{ font-size: 13px; color: #95a5a6; margin-top: 5px; }}
        .drift {{ font-size: 20px; font-weight: bold; color: {color}; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        th {{ background: #3498db; color: white; padding: 12px 15px; text-align: left; }}
        td {{ padding: 12px 15px; border-bottom: 1px solid #ecf0f1; }}
        tr:last-child td {{ border-bottom: none; }}
        .footer {{ margin-top: 40px; color: #95a5a6; font-size: 12px; text-align: center; }}
    </style>
</head>
<body>
    <h1>🛍️ StockSense — Rapport de Monitoring</h1>
    <p>Généré le <strong>{datetime.now().strftime('%d/%m/%Y à %H:%M')}</strong></p>

    <div class="status">{status}</div>

    <h2>📊 Métriques de performance</h2>
    <div class="grid">
        <div class="card">
            <h3>MAPE — Référence (entraînement)</h3>
            <div class="metric">{mape_ref:.1f}%</div>
            <div class="sub">Jan 2013 → Nov 2014</div>
        </div>
        <div class="card">
            <h3>MAPE — Courante (production)</h3>
            <div class="metric">{mape_cur:.1f}%</div>
            <div class="sub">Déc 2014 → Juil 2015</div>
        </div>
        <div class="card">
            <h3>MAE — Référence</h3>
            <div class="metric">{mae_ref:,.0f}</div>
            <div class="sub">unités / magasin / jour</div>
        </div>
        <div class="card">
            <h3>MAE — Courante</h3>
            <div class="metric">{mae_cur:,.0f}</div>
            <div class="sub">unités / magasin / jour</div>
        </div>
    </div>

    <h2>🔍 Analyse du drift</h2>
    <table>
        <tr><th>Métrique</th><th>Référence</th><th>Courante</th><th>Drift</th></tr>
        <tr>
            <td>MAPE</td>
            <td>{mape_ref:.1f}%</td>
            <td>{mape_cur:.1f}%</td>
            <td><span class="drift">{drift:+.1f}%</span></td>
        </tr>
        <tr>
            <td>MAE</td>
            <td>{mae_ref:,.0f}</td>
            <td>{mae_cur:,.0f}</td>
            <td><span class="drift">{mae_cur - mae_ref:+,.0f}</span></td>
        </tr>
        <tr>
            <td>Biais</td>
            <td>{bias_ref:+.0f}</td>
            <td>{bias_cur:+.0f}</td>
            <td><span class="drift">{bias_cur - bias_ref:+.0f}</span></td>
        </tr>
    </table>

    <h2>💡 Interprétation</h2>
    <div class="card">
        <p>Le seuil d'alerte est fixé à <strong>+5% de drift</strong> sur le MAPE.
        Un drift de <strong>{drift:+.1f}%</strong> a été détecté entre la période de référence
        et la période courante.</p>
        <p>{'Le modèle se dégrade significativement. Lancer <code>python retrain/retrain.py</code> pour réentraîner.' if drift > 5 else 'Le modèle reste dans les limites acceptables. Aucune action requise.'}</p>
    </div>

    <div class="footer">
        StockSense Demand Forecasting · Rapport généré automatiquement · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ Rapport généré : {report_path}")
    return report_path

def print_summary(reference, current):
    """Affiche un résumé des métriques dans la console."""
    from sklearn.metrics import mean_absolute_percentage_error

    mape_ref = mean_absolute_percentage_error(
        reference['Sales'], reference['prediction']) * 100
    mape_cur = mean_absolute_percentage_error(
        current['Sales'], current['prediction']) * 100

    drift = mape_cur - mape_ref

    print("\n" + "="*50)
    print("  RAPPORT DE MONITORING — StockSense")
    print("="*50)
    print(f"  MAPE référence  : {mape_ref:.1f}%")
    print(f"  MAPE courante   : {mape_cur:.1f}%")
    print(f"  Drift détecté   : {drift:+.1f}%")
    print("="*50)

    if drift > 5:
        print("  ⚠️  ALERTE : dégradation > 5% détectée.")
        print("  → Réentraînement recommandé.")
        print("  → Lancer : python retrain/retrain.py")
    else:
        print("  ✅ Modèle stable — pas de réentraînement nécessaire.")
    print("="*50)

if __name__ == '__main__':
    print("📊 Chargement des données...")
    reference, current = load_and_prepare()
    print_summary(reference, current)
    print("\n📄 Génération du rapport HTML...")
    generate_report(reference, current)