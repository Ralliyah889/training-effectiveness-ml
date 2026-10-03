# TrainIQ — Enterprise Training Intelligence & Analytics Platform

### B.Tech CSE Semester V Machine Learning Case Study (Case Study No. 88)
**Project Title:** Training Effectiveness Analysis Using Machine Learning  
**Subtitle:** Predicting Employee Post-Training Outcomes Based on Employee and Training Characteristics  
**Platform Name:** TrainIQ

---

## ⚡ Executive Overview

**TrainIQ** is a SaaS-grade People Analytics & Machine Learning application built to help HR leaders, talent development heads, and organizational psychologists understand the relationship between workforce characteristics, training investments, and post-training outcomes.

Built with an enterprise design system, TrainIQ combines:
1. **Mathematical Rigor:** Strict prevention of data leakage, stratified splitting, and balanced model evaluation.
2. **Interactive Visual Analytics:** Real-time multi-dimensional Plotly charts and cross-functional workforce segment filtering.
3. **Real-Time Inference Engine:** Instant post-training outcome estimation powered by trained Scikit-Learn pipelines.
4. **Model Governance & Auditing:** Transparent confusion matrix exploration, feature importance breakdowns, and academic limitations auditing.

---

## 🎨 Enterprise Design System

The TrainIQ user interface adheres to modern enterprise design principles:
* **Color Palette:**
  * Deep Navy: `#101828` (Sidebar & Branding)
  * Primary Blue: `#3563E9` (Key Actions & Accents)
  * Electric Blue: `#5C83FF` (Highlights & Badges)
  * Off-White: `#F8FAFC` (Canvas Background)
  * Success Green: `#20A779` (Positive Outcomes & Health)
  * Warning Amber: `#F3AA38` (Attention Indicators)
  * Lavender Accent: `#8B7CFF`
* **Typography:** Plus Jakarta Sans & Inter typography hierarchy.
* **Containers:** Glassmorphic navigation, soft shadow elevation (`0 4px 20px rgba(16, 24, 40, 0.04)`), rounded card surfaces, and responsive grids.

---

## 📱 Application Modules & Pages

| Page | Key Capabilities |
| :--- | :--- |
| **📊 Overview Dashboard** | Executive hero banner, 4 KPI metric cards, target distribution donut chart, training frequency analysis, department success rates, and automated dataset insights. |
| **📈 Training Analytics** | Multi-dimensional filter bar (Department, Job Role, Business Travel, Outcome), experience box plots, satisfaction distributions, and interactive correlation matrix. |
| **⚡ Employee Prediction** | Two-column layout with 3 structured input sections (Demographic Profile, Training History & Experience, Workplace Engagement) and a real-time prediction card with confidence bars and governance disclaimers. |
| **🧠 Model Insights** | Algorithm benchmark comparison (Logistic Regression, Decision Tree, Random Forest), interactive confusion matrix heatmaps, feature importance rankings, and class imbalance deep dive. |
| **🗄️ Dataset Explorer** | Data quality KPI summary (1,470 records, 35 columns, 0 missing values), interactive column multi-selector, pagination control, and full feature dictionary. |
| **ℹ️ About Project** | Case Study No. 88 specification, end-to-end ML architecture diagram, and technology stack badges. |

---

## 🔬 Machine Learning Architecture

```
Raw IBM HR Dataset (1,470 Records)
              │
              ▼
Data Preprocessing & Data Leakage Elimination
(Dropped: PercentSalaryHike, EmployeeCount, StandardHours, Over18)
              │
              ▼
ColumnTransformer Pipeline
├── Numerical Features: Median Imputer + StandardScaler
└── Categorical Features: Constant Imputer + OneHotEncoder
              │
              ▼
Stratified Train-Test Split (80% Train / 20% Test)
              │
              ▼
Multi-Model Training (Balanced Class Weights)
├── 1. Logistic Regression (Linear baseline)
├── 2. Decision Tree Classifier (Rule-based interpretability)
└── 3. Random Forest Classifier (Ensemble bagging)
              │
              ▼
Evaluation & Serialization (Joblib Pipeline)
              │
              ▼
TrainIQ Enterprise SaaS Interface (Streamlit)
```

### Verified Experimental Benchmark (Holdout Test Set = 294 Samples)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.5034 | 0.1203 | 0.3556 | 0.1798 | 0.4597 |
| **Decision Tree** | 0.6020 | 0.1250 | 0.2667 | 0.1702 | 0.4907 |
| **Random Forest** | **0.8435** | 0.0000 | 0.0000 | 0.0000 | 0.4900 |

> **Critical Academic Note:** `PercentSalaryHike` was explicitly eliminated to prevent artificial data leakage. On the non-leaked feature space, Random Forest optimizes majority class accuracy (~84.35%), whereas Logistic Regression with balanced weights captures ~35.6% recall at the cost of precision.

---

## 🚀 How to Run the Application

### 1. Environment Setup
```bash
# Clone or navigate to the project directory
cd "/Users/ralliyahrushdahep/Desktop/ML Exam/training-effectiveness-ml"

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Retrain Models
```bash
cd src
python3 data_preprocessing.py
python3 train_models.py
python3 evaluate_models.py
cd ..
```

### 3. Launch TrainIQ
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ Technology Stack
* **Language:** Python 3.14
* **App Framework:** Streamlit 1.43+
* **Machine Learning:** Scikit-Learn (Pipelines, ColumnTransformer, Ensembles)
* **Visualization:** Plotly Express & Plotly Graph Objects
* **Data Processing:** Pandas, NumPy
* **Serialization:** Joblib
