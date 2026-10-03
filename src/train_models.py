import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data, preprocess_and_split
import joblib
import os

def train_models():
    """Trains classification models and saves the best one."""
    print("Loading and preprocessing data...")
    df = load_data('../data/training_data.csv')
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = preprocess_and_split(df)
    
    # Transform training data
    X_train_transformed = preprocessor.transform(X_train)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    }
    
    trained_models = {}
    
    # Train each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_transformed, y_train)
        trained_models[name] = model
        # Save each model
        joblib.dump(model, f'../models/{name.replace(" ", "_").lower()}_model.pkl')
    
    # Final model is Random Forest for Streamlit App default
    joblib.dump(trained_models['Random Forest'], '../models/final_model.pkl')
    print("All models trained and saved to models/ directory.")

if __name__ == "__main__":
    train_models()
