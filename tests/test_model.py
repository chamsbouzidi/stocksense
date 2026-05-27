"""
test_model.py — Tests basiques du modèle StockSense
Usage : pytest tests/
"""

import joblib
import numpy as np
import os

MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.pkl')

def test_model_loads():
    """Le modèle doit se charger sans erreur."""
    model = joblib.load(MODEL_PATH)
    assert model is not None

def test_model_predicts():
    """Le modèle doit retourner une prédiction numérique positive."""
    model = joblib.load(MODEL_PATH)
    sample = np.array([[1, 0, 1, 0, 0, 0, 0, 1270,
                        2014, 6, 24, 15, 0, 0, 0, 0, 1, 0]])
    prediction = model.predict(sample)
    assert len(prediction) == 1
    assert prediction[0] > 0

def test_prediction_range():
    """La prédiction doit être dans une fourchette réaliste (0-50 000)."""
    model = joblib.load(MODEL_PATH)
    sample = np.array([[1, 0, 1, 0, 0, 0, 0, 1270,
                        2014, 6, 24, 15, 0, 0, 0, 0, 1, 0]])
    prediction = model.predict(sample)
    assert 0 < prediction[0] < 50000
