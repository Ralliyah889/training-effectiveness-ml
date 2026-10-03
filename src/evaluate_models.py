import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if os.path.join(BASE_DIR, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from data_preprocessing import load_data, preprocess_and_split

def get_model_evaluation_data():
    """
    Evaluates all trained models on the test split and returns:
    1. Summary DataFrame of metrics
    2. Dictionary of Confusion Matrices
    3. Dictionary of Classification Reports
    4. Feature Importances for Tree Models
    5. Test set stats (X_test shape, class counts)
    """
    csv_path = os.path.join(BASE_DIR, 'data', 'training_data.csv')
    df = load_data(csv_path)
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = preprocess_and_split(df)
    
    X_test_transformed = preprocessor.transform(X_test)
    
    model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    results = []
    cms = {}
    reports = {}
    models_dict = {}
    
    for name in model_names:
        file_name = name.replace(" ", "_").lower() + "_model.pkl"
        model_path = os.path.join(BASE_DIR, 'models', file_name)
        if not os.path.exists(model_path):
            continue
            
        model = joblib.load(model_path)
        models_dict[name] = model
        
        y_pred = model.predict(X_test_transformed)
        y_prob = model.predict_proba(X_test_transformed)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else np.nan
        
        cm = confusion_matrix(y_test, y_pred)
        cms[name] = cm
        reports[name] = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4),
            'ROC-AUC': round(roc, 4) if not np.isnan(roc) else 'N/A'
        })
        
    results_df = pd.DataFrame(results)
    
    # Extract feature names from preprocessor
    feature_names = []
    try:
        # Numeric features
        feature_names.extend(num_cols)
        # Categorical one-hot encoded features
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_encoded_names = cat_encoder.get_feature_names_out(cat_cols)
        feature_names.extend(cat_encoded_names)
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(X_test_transformed.shape[1])]
        
    rf_importances = models_dict.get('Random Forest').feature_importances_ if 'Random Forest' in models_dict else None
    
    return {
        "metrics_df": results_df,
        "confusion_matrices": cms,
        "classification_reports": reports,
        "feature_names": feature_names,
        "rf_importances": rf_importances,
        "test_size": len(y_test),
        "test_class_distribution": pd.Series(y_test).value_counts().to_dict()
    }

if __name__ == "__main__":
    data = get_model_evaluation_data()
    print("Evaluation Results:")
    print(data["metrics_df"])
