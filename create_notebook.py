import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# --- Kernel Metadata ---
nb.metadata['kernelspec'] = {
    'display_name': 'Python (training-ml)',
    'language': 'python',
    'name': 'training-ml'
}
nb.metadata['language_info'] = {
    'name': 'python',
    'version': '3.14.0'
}

# --- Cells ---

nb.cells.append(nbf.v4.new_markdown_cell("""\
# Training Effectiveness Analysis Using Machine Learning
**Case Study No. 88 — B.Tech CSE Semester V**

This notebook contains the complete end-to-end ML pipeline:
1. Data Loading & Exploration
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Model Training & Evaluation
5. Feature Importance
6. Error Analysis
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully.")
"""))

# --- Data Loading ---
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 1. Data Loading & Initial Exploration

**Dataset:** IBM HR Analytics Employee Attrition & Performance  
**Source:** Kaggle (via GitHub mirror)  
**Records:** 1470 | **Features:** 35
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
df = pd.read_csv('../data/training_data.csv')
print(f"Dataset Shape: {df.shape}")
print(f"Missing Values: {df.isnull().sum().sum()}")
print(f"Duplicate Rows: {df.duplicated().sum()}")
df.head()
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
df.info()
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
df.describe()
"""))

# --- Target Variable ---
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 2. Target Variable Construction

The dataset does not have a direct "Post-Training Outcome" column.  
We use `PerformanceRating` as a proxy:
- **4 (Outstanding)** → `Successful` (1)
- **3 (Excellent / baseline)** → `Needs Improvement` (0)

> **Note:** `PercentSalaryHike` is perfectly correlated with `PerformanceRating` in this synthetic dataset and is dropped to **prevent data leakage**.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
print("PerformanceRating distribution:")
print(df['PerformanceRating'].value_counts())

# Create binary target
df['Outcome'] = df['PerformanceRating'].apply(lambda x: 1 if x == 4 else 0)
print(f"\\nOutcome distribution:\\n{df['Outcome'].value_counts()}")
print(f"\\nClass imbalance ratio: {df['Outcome'].value_counts()[0] / df['Outcome'].value_counts()[1]:.2f}:1")
"""))

# --- EDA ---
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 3. Exploratory Data Analysis (EDA)
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# 1. Outcome Distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='Outcome', data=df, palette='viridis', ax=ax)
ax.set_xticklabels(['Needs Improvement (0)', 'Successful (1)'])
ax.set_title('Distribution of Post-Training Outcomes')
ax.set_ylabel('Count')
plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# 2. Training Times Last Year vs Outcome
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x='TrainingTimesLastYear', hue='Outcome', data=df, palette='coolwarm', ax=ax)
ax.set_title('Training Sessions Last Year vs Outcome')
ax.set_xlabel('Number of Training Sessions')
ax.set_ylabel('Count')
ax.legend(title='Outcome', labels=['Needs Improvement', 'Successful'])
plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# 3. Department vs Outcome
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x='Department', hue='Outcome', data=df, palette='Set2', ax=ax)
ax.set_title('Department vs Outcome')
ax.legend(title='Outcome', labels=['Needs Improvement', 'Successful'])
plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# 4. Job Satisfaction vs Outcome
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x='JobSatisfaction', hue='Outcome', data=df, palette='muted', ax=ax)
ax.set_title('Job Satisfaction Level vs Outcome')
ax.legend(title='Outcome', labels=['Needs Improvement', 'Successful'])
plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# 5. Total Working Years vs Outcome
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x='Outcome', y='TotalWorkingYears', data=df, palette='pastel', ax=ax)
ax.set_xticklabels(['Needs Improvement', 'Successful'])
ax.set_title('Total Working Years vs Outcome')
plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# 6. Age Distribution by Outcome
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Outcome', kde=True, palette='rocket', ax=ax)
ax.set_title('Age Distribution by Outcome')
plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# 7. Correlation Heatmap (numerical features)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()
"""))

# --- Preprocessing ---
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 4. Data Preprocessing

- Drop irrelevant columns (IDs, constants) and `PercentSalaryHike` (data leakage)
- Impute + Scale numerical features
- Impute + One-Hot Encode categorical features
- 80/20 stratified train-test split
- **Preprocessing is fitted only on training data** to avoid data leakage
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Drop columns
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours',
             'PercentSalaryHike', 'PerformanceRating']
drop_cols = [c for c in drop_cols if c in df.columns]
df_clean = df.drop(columns=drop_cols)

X = df_clean.drop(columns=['Outcome'])
y = df_clean['Outcome']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Build preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit on training data only
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print(f"Train set: {X_train_transformed.shape}")
print(f"Test set:  {X_test_transformed.shape}")
print(f"Train target distribution:\\n{y_train.value_counts()}")
print(f"Test target distribution:\\n{y_test.value_counts()}")
"""))

# --- Model Training ---
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 5. Model Training

We train three classifiers with `class_weight='balanced'` to handle the class imbalance:
1. **Logistic Regression** — linear baseline
2. **Decision Tree** — interpretable, non-linear
3. **Random Forest** — ensemble, robust
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_transformed, y_train)
    trained_models[name] = model
    print(f"✓ {name} trained.")
"""))

# --- Evaluation ---
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 6. Model Evaluation
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
results = []

for name, model in trained_models.items():
    y_pred = model.predict(X_test_transformed)
    y_prob = model.predict_proba(X_test_transformed)[:, 1]
    
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_prob)
    
    results.append({'Model': name, 'Accuracy': round(acc, 4),
                    'Precision': round(prec, 4), 'Recall': round(rec, 4),
                    'F1 Score': round(f1, 4), 'ROC-AUC': round(roc, 4)})

results_df = pd.DataFrame(results)
results_df
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = model.predict(X_test_transformed)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues',
                                            display_labels=['Needs Improv.', 'Successful'])
    ax.set_title(f'{name}')
plt.suptitle('Confusion Matrices', fontsize=14)
plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Detailed classification reports
for name, model in trained_models.items():
    y_pred = model.predict(X_test_transformed)
    print(f"\\n{'='*50}")
    print(f"{name}")
    print('='*50)
    print(classification_report(y_test, y_pred,
          target_names=['Needs Improvement', 'Successful'], zero_division=0))
"""))

# --- Feature Importance ---
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 7. Feature Importance

> **Disclaimer:** Feature importance shows which features were important for the model's predictions.  
> It does **not** prove causation.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Random Forest feature importance
rf = trained_models['Random Forest']

# Get feature names after one-hot encoding
ohe_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist()
all_features = numeric_features + ohe_features

importances = rf.feature_importances_
feat_imp = pd.DataFrame({'Feature': all_features, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax)
ax.set_title('Top 15 Feature Importances (Random Forest)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Logistic Regression coefficients (top features)
lr = trained_models['Logistic Regression']
lr_coefs = pd.DataFrame({'Feature': all_features, 'Coefficient': lr.coef_[0]})
lr_coefs['Abs_Coeff'] = lr_coefs['Coefficient'].abs()
lr_coefs = lr_coefs.sort_values('Abs_Coeff', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=lr_coefs, palette='coolwarm', ax=ax)
ax.set_title('Top 15 Logistic Regression Coefficients')
plt.tight_layout()
plt.show()
"""))

# --- Error Analysis ---
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 8. Error Analysis

### Key Observations:
- The dataset is **highly imbalanced** (~85% class 0, ~15% class 1).
- After removing `PercentSalaryHike`, the remaining features have very weak predictive power for `PerformanceRating`.
- **Random Forest** achieves high accuracy by defaulting to the majority class (0 recall for minority).
- **Logistic Regression** (with balanced weights) attempts to capture the minority class but at the cost of many false positives.

### Limitations:
1. The IBM dataset is **synthetic** — patterns may not reflect real organizational behaviour.
2. There is no explicit pre-training / post-training assessment score to measure a direct improvement delta.
3. `TrainingTimesLastYear` records frequency, not quality, duration, or relevance of training.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Misclassified records analysis (using Logistic Regression as it attempts minority prediction)
lr_model = trained_models['Logistic Regression']
y_pred_lr = lr_model.predict(X_test_transformed)

misclassified = X_test.copy()
misclassified['Actual'] = y_test.values
misclassified['Predicted'] = y_pred_lr

false_positives = misclassified[(misclassified['Actual'] == 0) & (misclassified['Predicted'] == 1)]
false_negatives = misclassified[(misclassified['Actual'] == 1) & (misclassified['Predicted'] == 0)]

print(f"False Positives (predicted Successful, actually Needs Improvement): {len(false_positives)}")
print(f"False Negatives (predicted Needs Improvement, actually Successful): {len(false_negatives)}")
print(f"Total Misclassified: {len(false_positives) + len(false_negatives)} / {len(y_test)}")
"""))

# --- Save Model ---
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 9. Save Models
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
import joblib
import os

os.makedirs('../models', exist_ok=True)
joblib.dump(preprocessor, '../models/preprocessing.pkl')
joblib.dump(trained_models['Random Forest'], '../models/final_model.pkl')

for name, model in trained_models.items():
    fname = name.replace(' ', '_').lower() + '_model.pkl'
    joblib.dump(model, f'../models/{fname}')

print("All models and preprocessor saved to ../models/")
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""\
## 10. Conclusion

- The ML pipeline is fully functional and reproducible.
- Due to the synthetic nature of the data and the removal of the leaky `PercentSalaryHike` feature, prediction performance is limited.
- This demonstrates the importance of having high-quality, real-world data with explicit training effectiveness measurements.
- The Streamlit app (`app.py`) provides an interactive dashboard for exploring results and making predictions.
"""))

# --- Write ---
out_path = '/Users/ralliyahrushdahep/Desktop/ML Exam/training-effectiveness-ml/notebooks/training_effectiveness_analysis.ipynb'
nbf.write(nb, out_path)
print(f"Notebook written to {out_path}")
