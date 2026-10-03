import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def load_data(filepath):
    """Loads the dataset from the given filepath."""
    return pd.read_csv(filepath)

def preprocess_and_split(df, target_col='Outcome', test_size=0.2, random_state=42):
    """
    Preprocesses the data and splits it into training and testing sets.
    Maps PerformanceRating to binary outcomes if it exists.
    """
    df = df.copy()
    
    # Check for duplicates and drop them
    df = df.drop_duplicates()
    
    # Define Target: Case Study 88 - Training Effectiveness
    # We map PerformanceRating: 4 -> 'Successful' (1), 3 -> 'Needs Improvement' (0)
    if 'PerformanceRating' in df.columns:
        df['Outcome'] = df['PerformanceRating'].apply(lambda x: 1 if x == 4 else 0)
        df = df.drop(columns=['PerformanceRating'])
    
    # Drop columns that are irrelevant (like IDs) or cause data leakage (PercentSalaryHike perfectly predicts PerformanceRating in this dataset)
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'PercentSalaryHike']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Build preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Fit preprocessor on training data
    preprocessor.fit(X_train)
    
    # Save the fitted preprocessor
    os.makedirs('../models', exist_ok=True)
    joblib.dump(preprocessor, '../models/preprocessing.pkl')
    
    return X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features

if __name__ == "__main__":
    df = load_data('../data/training_data.csv')
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = preprocess_and_split(df)
    print(f"Data preprocessing completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
