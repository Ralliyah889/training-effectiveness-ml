import pandas as pd
import numpy as np
import joblib
import os

# Robust path resolution relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_pipeline():
    """Loads the preprocessor and final Random Forest model."""
    prep_path = os.path.join(MODELS_DIR, 'preprocessing.pkl')
    model_path = os.path.join(MODELS_DIR, 'final_model.pkl')
    
    if not os.path.exists(prep_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model files not found in {MODELS_DIR}. Run train_models.py first.")
        
    preprocessor = joblib.load(prep_path)
    model = joblib.load(model_path)
    return preprocessor, model

def load_all_models():
    """Loads all 3 trained models for comparison and inspection."""
    prep_path = os.path.join(MODELS_DIR, 'preprocessing.pkl')
    preprocessor = joblib.load(prep_path)
    
    models = {}
    for name, filename in [
        ('Logistic Regression', 'logistic_regression_model.pkl'),
        ('Decision Tree', 'decision_tree_model.pkl'),
        ('Random Forest', 'random_forest_model.pkl')
    ]:
        p = os.path.join(MODELS_DIR, filename)
        if os.path.exists(p):
            models[name] = joblib.load(p)
            
    return preprocessor, models

def predict_single_employee(input_data_dict: dict, model_name: str = "Random Forest"):
    """
    Takes an employee feature dictionary, runs it through the full preprocessor,
    and returns outcome, confidence, probability breakdown, and feature details.
    """
    preprocessor, models = load_all_models()
    model = models.get(model_name, models.get('Random Forest'))
    
    input_df = pd.DataFrame([input_data_dict])
    
    # Drop any excluded columns if accidentally present
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'PercentSalaryHike', 'PerformanceRating', 'Outcome']
    input_df = input_df.drop(columns=[c for c in drop_cols if c in input_df.columns])
    
    # Transform using ColumnTransformer
    transformed = preprocessor.transform(input_df)
    
    prediction = int(model.predict(transformed)[0])
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(transformed)[0]
        prob_0 = float(proba[0])
        prob_1 = float(proba[1]) if len(proba) > 1 else 0.0
        confidence = float(proba[prediction])
    else:
        prob_0 = 1.0 if prediction == 0 else 0.0
        prob_1 = 1.0 if prediction == 1 else 0.0
        confidence = 1.0
        
    outcome_label = "Successful" if prediction == 1 else "Needs Improvement"
    
    return {
        "prediction": prediction,
        "outcome_label": outcome_label,
        "confidence": confidence,
        "probabilities": {
            "Needs Improvement (Class 0)": prob_0,
            "Successful (Class 1)": prob_1
        },
        "model_used": model_name
    }
