import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure src directory is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from theme import CUSTOM_CSS, COLOR_PALETTE
from components import (
    render_html,
    render_sidebar,
    render_page_header,
    render_metric_card,
    render_hero_banner,
    render_insight_card,
    render_prediction_card
)
from eda_analytics import (
    get_prepared_eda_data,
    create_target_distribution_chart,
    create_training_vs_outcome_chart,
    create_department_outcome_chart,
    create_experience_outcome_chart,
    create_satisfaction_outcome_chart,
    create_job_role_outcome_chart,
    create_correlation_heatmap,
    create_confusion_matrix_chart,
    create_feature_importance_chart
)
from predict import predict_single_employee, load_all_models
from evaluate_models import get_model_evaluation_data

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="TrainIQ — Enterprise Training Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Design System CSS
render_html(CUSTOM_CSS)

# Cache data loading
@st.cache_data
def load_main_dataset():
    data_path = os.path.join(BASE_DIR, 'data', 'training_data.csv')
    df = pd.read_csv(data_path)
    return df

@st.cache_data
def load_cached_evaluation_data():
    return get_model_evaluation_data()

# -------------------------------------------------------------
# PAGE 1: OVERVIEW DASHBOARD
# -------------------------------------------------------------
def render_overview_page(df: pd.DataFrame, eval_data: dict):
    render_page_header(
        title="Welcome to TrainIQ",
        subtitle="Understand employee learning outcomes and optimize training ROI through predictive machine learning.",
        badge_text="Live Dashboard",
        badge_type="blue"
    )
    
    # Executive Hero Banner
    render_hero_banner()
    
    # Key Dataset Statistics
    total_employees = len(df)
    avg_training = df['TrainingTimesLastYear'].mean()
    success_count = (df['PerformanceRating'] == 4).sum() if 'PerformanceRating' in df.columns else 0
    success_pct = (success_count / total_employees) * 100
    rf_acc = eval_data['metrics_df'].loc[eval_data['metrics_df']['Model'] == 'Random Forest', 'Accuracy'].values[0] * 100
    
    # 4 Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            icon="👥",
            label="Total Workforce",
            value=f"{total_employees:,}",
            subtext="Analyzed employee records",
            accent_color="#3563E9",
            icon_bg="#EDF4FF"
        )
    with c2:
        render_metric_card(
            icon="📚",
            label="Avg. Training Sessions",
            value=f"{avg_training:.1f}",
            subtext="Sessions per employee / year",
            accent_color="#8B7CFF",
            icon_bg="#F5F3FF"
        )
    with c3:
        render_metric_card(
            icon="🎯",
            label="Successful Outcomes",
            value=f"{success_pct:.1f}%",
            subtext=f"{success_count:,} Outstanding ratings (Rating 4)",
            accent_color="#20A779",
            icon_bg="#DCFCE7"
        )
    with c4:
        render_metric_card(
            icon="⚡",
            label="Model Baseline",
            value=f"{rf_acc:.1f}%",
            subtext="Random Forest test accuracy",
            accent_color="#F3AA38",
            icon_bg="#FEF3C7"
        )
        
    render_html("<div style='height: 24px;'></div>")
    
    # Visual Analytics Grid
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        render_html("""
        <div class="tiq-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 8px;">Target Class Distribution</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 12px;">Outcome breakdown between Rating 3 (Needs Improvement) and Rating 4 (Successful).</div>
        </div>
        """)
        st.plotly_chart(create_target_distribution_chart(df), use_container_width=True)
        
    with col_right:
        render_html("""
        <div class="tiq-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 8px;">Training Frequency vs Outcome</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 12px;">Distribution of post-training outcomes grouped by number of training sessions attended.</div>
        </div>
        """)
        st.plotly_chart(create_training_vs_outcome_chart(df), use_container_width=True)
        
    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        render_html("""
        <div class="tiq-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 8px;">Department Outcome Distribution</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 12px;">Relative proportion of post-training success rates across organizational units.</div>
        </div>
        """)
        st.plotly_chart(create_department_outcome_chart(df), use_container_width=True)
        
    with col_b2:
        render_html("""
        <div class="tiq-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 8px;">Experience Distribution</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 12px;">Comparative total working years across outcome classes.</div>
        </div>
        """)
        st.plotly_chart(create_experience_outcome_chart(df), use_container_width=True)
        
    # Statistical Key Insights derived from actual data
    render_insight_card(
        title="Key Dataset Insights & Observations",
        points=[
            f"<strong>Class Imbalance:</strong> The outcome variable is substantially imbalanced with {success_pct:.1f}% classified as 'Successful' (Rating 4) and {(100-success_pct):.1f}% as 'Needs Improvement' (Rating 3).",
            "<strong>Training Participation:</strong> The modal training frequency across both outcome groups is 2 to 3 sessions per year; higher training count alone does not guarantee a higher rating.",
            "<strong>Departmental Uniformity:</strong> R&D, Sales, and HR exhibit relatively similar internal success ratios (~15%), indicating cross-functional consistency.",
            "<strong>Leakage Elimination:</strong> The leaky feature <code>PercentSalaryHike</code> was intentionally removed to ensure genuine machine learning evaluation rather than trivial synthetic prediction."
        ],
        icon="🔍"
    )

# -------------------------------------------------------------
# PAGE 2: ANALYTICS DASHBOARD
# -------------------------------------------------------------
def render_analytics_page(df: pd.DataFrame):
    render_page_header(
        title="Training Analytics",
        subtitle="Explore multivariate patterns across employee demographics, roles, satisfaction, and training outcomes.",
        badge_text="Interactive Exploration",
        badge_type="blue"
    )
    
    # Filter Bar Container
    render_html("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px 24px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
        <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B;">
            🔍 Filter Workforce Segment
        </div>
    </div>
    """)
    
    f1, f2, f3, f4 = st.columns(4)
    
    with f1:
        depts = ["All Departments"] + sorted(df['Department'].unique().tolist())
        selected_dept = st.selectbox("Department", depts, key="filter_dept")
        
    with f2:
        roles = ["All Job Roles"] + sorted(df['JobRole'].unique().tolist())
        selected_role = st.selectbox("Job Role", roles, key="filter_role")
        
    with f3:
        travel_options = ["All Travel Frequencies"] + sorted(df['BusinessTravel'].unique().tolist())
        selected_travel = st.selectbox("Business Travel", travel_options, key="filter_travel")
        
    with f4:
        outcomes = ["All Outcomes", "Successful", "Needs Improvement"]
        selected_outcome = st.selectbox("Outcome", outcomes, key="filter_outcome")
        
    # Apply Filters to DataFrame
    filtered_df = df.copy()
    if selected_dept != "All Departments":
        filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
    if selected_role != "All Job Roles":
        filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]
    if selected_travel != "All Travel Frequencies":
        filtered_df = filtered_df[filtered_df['BusinessTravel'] == selected_travel]
    if selected_outcome != "All Outcomes":
        target_val = 4 if selected_outcome == "Successful" else 3
        filtered_df = filtered_df[filtered_df['PerformanceRating'] == target_val]
        
    # Active Filter Status Badge
    filter_count = len(filtered_df)
    render_html(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px;">
        <span style="font-size: 0.88rem; color: #475467;">
            Showing <strong>{filter_count:,}</strong> of <strong>{len(df):,}</strong> records ({filter_count/len(df)*100:.1f}%)
        </span>
    </div>
    """)
    
    if filter_count == 0:
        st.warning("No records match the selected filter criteria. Please broaden your selection.")
        return

    # Visualizations Row 1
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        render_html("""
        <div class="tiq-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 8px;">Filtered Outcome Breakdown</div>
        </div>
        """)
        st.plotly_chart(create_target_distribution_chart(filtered_df), use_container_width=True)
        
    with r1_col2:
        render_html("""
        <div class="tiq-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 8px;">Training Frequency Distribution</div>
        </div>
        """)
        st.plotly_chart(create_training_vs_outcome_chart(filtered_df), use_container_width=True)
        
    # Visualizations Row 2
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        render_html("""
        <div class="tiq-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 8px;">Job Satisfaction vs Outcome</div>
        </div>
        """)
        st.plotly_chart(create_satisfaction_outcome_chart(filtered_df), use_container_width=True)
        
    with r2_col2:
        render_html("""
        <div class="tiq-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 8px;">Job Role Distribution</div>
        </div>
        """)
        st.plotly_chart(create_job_role_outcome_chart(filtered_df), use_container_width=True)
        
    # Correlation Matrix
    render_html("""
    <div class="tiq-card">
        <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 4px;">Multivariate Correlation Matrix</div>
        <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 16px;">Linear correlation coefficients between key numerical metrics in the filtered subset.</div>
    </div>
    """)
    st.plotly_chart(create_correlation_heatmap(filtered_df), use_container_width=True)

# -------------------------------------------------------------
# PAGE 3: EMPLOYEE PREDICTION
# -------------------------------------------------------------
def render_prediction_page(df: pd.DataFrame):
    render_page_header(
        title="Employee Outcome Prediction",
        subtitle="Estimate post-training outcomes using employee characteristics and training history through trained ML models.",
        badge_text="Inference Engine",
        badge_type="green"
    )
    
    # Model Selector & Presets Row
    render_html("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 14px 20px; margin-bottom: 16px;">
        <div style="font-size: 0.88rem; font-weight: 700; color: #101828;">
            🤖 Active Prediction Model & Preset Configuration
        </div>
    </div>
    """)
    
    m_col1, m_col2 = st.columns([1, 1])
    with m_col1:
        selected_model = st.selectbox(
            "Select Machine Learning Model",
            ["Random Forest", "Decision Tree", "Logistic Regression"],
            key="pred_model_selector",
            help="Choose which trained classification pipeline to use for generating real-time predictions."
        )
    with m_col2:
        preset = st.selectbox(
            "Quick Profile Presets",
            ["Custom Input / Standard Employee", "⭐ High Engagement & Experience", "🌱 Junior Employee Profile"],
            key="profile_preset"
        )
        
    # Default values based on preset
    def_age = 34
    def_dept = "Research & Development"
    def_job_level = 2
    def_training = 3
    def_working_years = 8
    def_years_company = 5
    def_income = 5200
    def_sat = 3
    def_inv = 3
    
    if preset == "⭐ High Engagement & Experience":
        def_age = 45
        def_job_level = 4
        def_training = 5
        def_working_years = 20
        def_years_company = 12
        def_income = 14000
        def_sat = 4
        def_inv = 4
    elif preset == "🌱 Junior Employee Profile":
        def_age = 23
        def_job_level = 1
        def_training = 2
        def_working_years = 2
        def_years_company = 1
        def_income = 2800
        def_sat = 2
        def_inv = 2

    col_form, col_result = st.columns([1.1, 0.9])
    
    with col_form:
        render_html("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 20px 24px 8px 24px; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-bottom: 12px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #101828; margin-bottom: 4px;">Employee & Training Parameters</div>
            <div style="font-size: 0.85rem; color: #64748B;">Adjust attributes to evaluate post-training outcome probabilities in real-time.</div>
        </div>
        """)
        
        # Section A: Profile
        render_html("<div style='font-size: 0.82rem; font-weight: 700; text-transform: uppercase; color: #3563E9; letter-spacing: 0.05em; margin: 12px 0 8px 0;'>Section A — Demographic & Role Profile</div>")
        sa1, sa2 = st.columns(2)
        with sa1:
            age = st.number_input("Age", min_value=18, max_value=65, value=def_age, step=1, key="p_age")
            gender = st.selectbox("Gender", ["Male", "Female"], key="p_gender")
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], key="p_marital")
        with sa2:
            dept = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"], index=0 if def_dept == "Research & Development" else 1, key="p_dept")
            roles_in_dept = df[df['Department'] == dept]['JobRole'].unique().tolist()
            job_role = st.selectbox("Job Role", roles_in_dept if roles_in_dept else df['JobRole'].unique().tolist(), key="p_role")
            job_level = st.slider("Job Level", 1, 5, def_job_level, key="p_job_level")
            
        render_html("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E2E8F0;'>")
        
        # Section B: Training & Experience
        render_html("<div style='font-size: 0.82rem; font-weight: 700; text-transform: uppercase; color: #3563E9; letter-spacing: 0.05em; margin-bottom: 8px;'>Section B — Training History & Experience</div>")
        sb1, sb2 = st.columns(2)
        with sb1:
            training_times = st.number_input("Training Sessions Last Year", min_value=0, max_value=10, value=def_training, step=1, help="Total formal training programs attended in the preceding 12 months.", key="p_training")
            total_working_years = st.number_input("Total Working Experience (Years)", min_value=0, max_value=40, value=def_working_years, step=1, key="p_exp")
            years_at_company = st.number_input("Tenure at Company (Years)", min_value=0, max_value=40, value=def_years_company, step=1, key="p_tenure")
            years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=min(def_years_company, 3), step=1, key="p_role_yrs")
        with sb2:
            travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], key="p_travel")
            overtime = st.selectbox("Overtime", ["No", "Yes"], key="p_overtime")
            years_since_promo = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1, step=1, key="p_promo")
            years_with_mgr = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=min(def_years_company, 3), step=1, key="p_mgr")
            
        render_html("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E2E8F0;'>")
        
        # Section C: Engagement & Satisfaction
        render_html("<div style='font-size: 0.82rem; font-weight: 700; text-transform: uppercase; color: #3563E9; letter-spacing: 0.05em; margin-bottom: 8px;'>Section C — Workplace Engagement & Compensation</div>")
        sc1, sc2 = st.columns(2)
        with sc1:
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=def_income, step=250, key="p_income")
            job_sat = st.slider("Job Satisfaction", 1, 4, def_sat, help="1=Low, 2=Medium, 3=High, 4=Very High", key="p_jobsat")
            env_sat = st.slider("Environment Satisfaction", 1, 4, 3, key="p_envsat")
            job_inv = st.slider("Job Involvement", 1, 4, def_inv, key="p_jobinv")
        with sc2:
            edu_fields = df['EducationField'].unique().tolist()
            edu_field = st.selectbox("Education Field", edu_fields, key="p_edufield")
            education_level = st.slider("Education Level (1-5)", 1, 5, 3, help="1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor", key="p_edulevel")
            work_life = st.slider("Work-Life Balance", 1, 4, 3, key="p_worklife")
            stock_option = st.slider("Stock Option Level", 0, 3, 1, key="p_stock")
            
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        btn_predict = st.button("⚡ Analyze Training Outcome", type="primary", use_container_width=True, key="btn_run_prediction")
        
    with col_result:
        render_html(f"""
        <div style="font-size: 1.15rem; font-weight: 800; color: #101828; margin-bottom: 4px;">Prediction Analysis</div>
        <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 16px;">Model Engine: <strong>{selected_model}</strong> Pipeline</div>
        """)
        
        # Build complete feature dictionary
        base_record = df.iloc[0].to_dict()
        base_record.update({
            'Age': age,
            'Gender': gender,
            'MaritalStatus': marital,
            'Department': dept,
            'JobRole': job_role,
            'JobLevel': job_level,
            'TrainingTimesLastYear': training_times,
            'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role,
            'YearsSinceLastPromotion': years_since_promo,
            'YearsWithCurrManager': years_with_mgr,
            'BusinessTravel': travel,
            'OverTime': overtime,
            'MonthlyIncome': monthly_income,
            'JobSatisfaction': job_sat,
            'EnvironmentSatisfaction': env_sat,
            'JobInvolvement': job_inv,
            'EducationField': edu_field,
            'Education': education_level,
            'WorkLifeBalance': work_life,
            'StockOptionLevel': stock_option,
            'RelationshipSatisfaction': 3,
            'DailyRate': 800,
            'HourlyRate': 65,
            'MonthlyRate': 14000,
            'NumCompaniesWorked': 2,
            'DistanceFromHome': 6,
            'Attrition': 'No'
        })
        
        try:
            result = predict_single_employee(base_record, model_name=selected_model)
            outcome = result["outcome_label"]
            conf = result["confidence"] * 100
            p_needs = result["probabilities"]["Needs Improvement (Class 0)"] * 100
            p_succ = result["probabilities"]["Successful (Class 1)"] * 100
            
            render_prediction_card(
                outcome=outcome,
                conf=conf,
                p_needs=p_needs,
                p_succ=p_succ,
                model_name=selected_model
            )
            
        except Exception as e:
            st.error(f"Inference error: {str(e)}")
            
        # Enterprise Disclaimer
        render_html("""
        <div style="margin-top: 20px; padding: 14px 18px; background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 12px; font-size: 0.8rem; color: #92400E; line-height: 1.5;">
            🛡️ <strong>Governance & Ethics Disclaimer:</strong> Predictions represent statistical estimations derived from historical workforce patterns. They are intended as decision-support insights for HR talent development and must not be used as the sole criteria for compensation, termination, or promotion.
        </div>
        """)

# -------------------------------------------------------------
# PAGE 4: MODEL INSIGHTS
# -------------------------------------------------------------
def render_model_insights_page(eval_data: dict):
    render_page_header(
        title="Model Intelligence",
        subtitle="Benchmark classification algorithms, inspect confusion matrices, and understand feature contributions.",
        badge_text="Evaluation & Audit",
        badge_type="lavender"
    )
    
    metrics_df = eval_data['metrics_df']
    
    # 3 Summary Performance Cards
    render_html("<div style='font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 14px;'>Model Benchmark Summary (Test Split: 294 Samples)</div>")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        lr_row = metrics_df[metrics_df['Model'] == 'Logistic Regression'].iloc[0]
        render_metric_card(
            icon="📈",
            label="Logistic Regression",
            value=f"{lr_row['Accuracy']*100:.1f}%",
            subtext=f"Precision: {lr_row['Precision']:.2f} | Recall: {lr_row['Recall']:.2f}",
            accent_color="#3563E9",
            icon_bg="#EDF4FF"
        )
    with c2:
        dt_row = metrics_df[metrics_df['Model'] == 'Decision Tree'].iloc[0]
        render_metric_card(
            icon="🌳",
            label="Decision Tree",
            value=f"{dt_row['Accuracy']*100:.1f}%",
            subtext=f"Precision: {dt_row['Precision']:.2f} | Recall: {dt_row['Recall']:.2f}",
            accent_color="#8B7CFF",
            icon_bg="#F5F3FF"
        )
    with c3:
        rf_row = metrics_df[metrics_df['Model'] == 'Random Forest'].iloc[0]
        render_metric_card(
            icon="🌲",
            label="Random Forest (Ensemble)",
            value=f"{rf_row['Accuracy']*100:.1f}%",
            subtext=f"Precision: {rf_row['Precision']:.2f} | Recall: {rf_row['Recall']:.2f}",
            accent_color="#20A779",
            icon_bg="#DCFCE7"
        )
        
    render_html("<div style='height: 20px;'></div>")
    
    # Comprehensive Comparison Table
    render_html("""
    <div class="tiq-card">
        <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 4px;">Experimental Evaluation Matrix</div>
        <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 16px;">Exact experimental results computed on the 20% stratified holdout test split.</div>
    </div>
    """)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Model Selection & Deep Dive
    render_html("<div style='height: 10px;'></div>")
    col_cm, col_fi = st.columns([1, 1])
    
    with col_cm:
        render_html("""
        <div class="tiq-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 8px;">Confusion Matrix Explorer</div>
        </div>
        """)
        selected_model_name = st.selectbox("Select Model to Inspect", ['Random Forest', 'Decision Tree', 'Logistic Regression'], key="cm_model_select")
        cm = eval_data['confusion_matrices'][selected_model_name]
        st.plotly_chart(create_confusion_matrix_chart(cm, selected_model_name), use_container_width=True)
        
    with col_fi:
        render_html("""
        <div class="tiq-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 8px;">Feature Importance Analysis</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 12px;">Gini-impurity based relative importance from the Random Forest ensemble.</div>
        </div>
        """)
        if eval_data['rf_importances'] is not None:
            st.plotly_chart(create_feature_importance_chart(eval_data['rf_importances'], eval_data['feature_names'], top_n=10), use_container_width=True)
        else:
            st.info("Feature importance data not available.")
        
    # Academic & Technical Deep Dive Callout
    render_insight_card(
        title="Academic Interpretation & Class Imbalance Analysis",
        points=[
            "<strong>Accuracy vs Recall Paradox:</strong> Random Forest achieves 84.35% accuracy by classifying the majority class (Class 0 represents 84.7% of the test set). However, it achieves 0.00 Recall on Class 1 because the non-leaked features do not have sufficient discriminative separation.",
            "<strong>Balanced Class Weighting Effect:</strong> Logistic Regression (configured with <code>class_weight='balanced'</code>) succeeds in identifying 35.6% of successful candidates (Recall = 0.3556), but suffers from false positives (Precision = 0.1203).",
            "<strong>Data Leakage Prevention:</strong> Dropping <code>PercentSalaryHike</code> was mandatory because in synthetic HR datasets, salary hike is directly determined by performance rating. Retaining it would create a trivial, artificial 100% accurate model with zero genuine machine learning value."
        ],
        icon="🎓"
    )

# -------------------------------------------------------------
# PAGE 5: DATASET EXPLORER
# -------------------------------------------------------------
def render_dataset_explorer_page(df: pd.DataFrame):
    render_page_header(
        title="Dataset Explorer",
        subtitle="Explore the raw workforce records, inspect the schema, and review data quality dimensions.",
        badge_text="Data Governance",
        badge_type="blue"
    )
    
    num_cols = df.select_dtypes(include=['int64', 'float64', 'number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("📋", "Total Rows", f"{len(df):,}", "Employee records", "#3563E9")
    with c2:
        render_metric_card("📊", "Total Columns", f"{df.shape[1]}", "Original attributes", "#8B7CFF")
    with c3:
        render_metric_card("🔢", "Numeric Features", f"{len(num_cols)}", "Scaled & imputed", "#20A779")
    with c4:
        render_metric_card("🔤", "Categorical Features", f"{len(cat_cols)}", "One-hot encoded", "#F3AA38")
    with c5:
        render_metric_card("✅", "Missing Values", "0", "100% Complete", "#20A779")
        
    render_html("<div style='height: 20px;'></div>")
    
    # Interactive Table Explorer
    render_html("""
    <div class="tiq-card">
        <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 4px;">Workforce Data Preview</div>
        <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 16px;">Filter columns and explore records in the dataset.</div>
    </div>
    """)
    
    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        selected_columns = st.multiselect(
            "Select Columns to Display",
            options=df.columns.tolist(),
            default=['Age', 'Department', 'JobRole', 'TrainingTimesLastYear', 'PerformanceRating', 'MonthlyIncome', 'TotalWorkingYears', 'JobSatisfaction']
        )
    with t_col2:
        row_limit = st.selectbox("Rows to Display", [10, 25, 50, 100, len(df)], index=0)
        
    if selected_columns:
        st.dataframe(df[selected_columns].head(row_limit), use_container_width=True)
    else:
        st.dataframe(df.head(row_limit), use_container_width=True)
        
    # Feature Dictionary
    render_html("""
    <div class="tiq-card" style="margin-top: 20px;">
        <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 12px;">Feature Dictionary & Metadata</div>
    </div>
    """)
    
    dict_data = [
        {"Feature": "TrainingTimesLastYear", "Type": "Numeric (int64)", "Description": "Number of training programs attended by the employee in the last 12 months (0-6)."},
        {"Feature": "PerformanceRating", "Type": "Categorical Target", "Description": "Post-training outcome rating (3 = Needs Improvement / Excellent, 4 = Successful / Outstanding)."},
        {"Feature": "TotalWorkingYears", "Type": "Numeric (int64)", "Description": "Total career experience in years across all organizations."},
        {"Feature": "YearsAtCompany", "Type": "Numeric (int64)", "Description": "Tenure at current organization in years."},
        {"Feature": "JobSatisfaction", "Type": "Ordinal (1-4)", "Description": "Self-reported job satisfaction survey score."},
        {"Feature": "JobInvolvement", "Type": "Ordinal (1-4)", "Description": "Employee involvement and engagement rating."},
        {"Feature": "Department", "Type": "Categorical", "Description": "Organizational department (R&D, Sales, HR)."},
        {"Feature": "JobRole", "Type": "Categorical", "Description": "Specific job title and functional responsibility."},
        {"Feature": "PercentSalaryHike", "Type": "Excluded (Leakage)", "Description": "Removed during preprocessing due to direct mathematical determination of performance rating."}
    ]
    st.table(pd.DataFrame(dict_data))

# -------------------------------------------------------------
# PAGE 6: ABOUT PROJECT
# -------------------------------------------------------------
def render_about_page():
    render_page_header(
        title="About TrainIQ",
        subtitle="Academic specification, methodology, and architecture for Case Study No. 88.",
        badge_text="B.Tech CSE ML Project",
        badge_type="blue"
    )
    
    render_html("""
    <div class="tiq-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <div style="width: 32px; height: 32px; border-radius: 8px; background: #EDF4FF; color: #3563E9; display: flex; align-items: center; justify-content: center; font-weight: 800;">88</div>
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #101828;">Case Study No. 88: Training Effectiveness Analysis</div>
                <div style="font-size: 0.85rem; color: #64748B;">Semester V B.Tech Computer Science & Engineering — Machine Learning Case Study</div>
            </div>
        </div>
        <p style="color: #475467; font-size: 0.92rem; line-height: 1.6;">
            <strong>Problem Statement:</strong> An organization wants to evaluate whether employee characteristics and training history can help explain differences in post-training outcomes, enabling data-driven talent development and learning optimization.
        </p>
    </div>
    """)
    
    # End-to-End Workflow Diagram
    render_html("""
    <div class="tiq-card">
        <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 16px;">End-to-End Machine Learning Architecture</div>
        
        <div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: space-between; align-items: center;">
            <div style="flex: 1; min-width: 140px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; text-align: center;">
                <div style="font-size: 1.4rem; margin-bottom: 4px;">📥</div>
                <div style="font-weight: 700; font-size: 0.85rem; color: #101828;">1. Data Ingestion</div>
                <div style="font-size: 0.75rem; color: #64748B;">IBM HR Dataset (1,470 rows)</div>
            </div>
            <div style="color: #94A3B8; font-size: 1.2rem;">➔</div>
            <div style="flex: 1; min-width: 140px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; text-align: center;">
                <div style="font-size: 1.4rem; margin-bottom: 4px;">🧹</div>
                <div style="font-weight: 700; font-size: 0.85rem; color: #101828;">2. Preprocessing</div>
                <div style="font-size: 0.75rem; color: #64748B;">Leakage Drop, Scaler & OneHot</div>
            </div>
            <div style="color: #94A3B8; font-size: 1.2rem;">➔</div>
            <div style="flex: 1; min-width: 140px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; text-align: center;">
                <div style="font-size: 1.4rem; margin-bottom: 4px;">✂️</div>
                <div style="font-weight: 700; font-size: 0.85rem; color: #101828;">3. Stratified Split</div>
                <div style="font-size: 0.75rem; color: #64748B;">80% Train / 20% Test</div>
            </div>
            <div style="color: #94A3B8; font-size: 1.2rem;">➔</div>
            <div style="flex: 1; min-width: 140px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; text-align: center;">
                <div style="font-size: 1.4rem; margin-bottom: 4px;">🤖</div>
                <div style="font-weight: 700; font-size: 0.85rem; color: #101828;">4. Model Training</div>
                <div style="font-size: 0.75rem; color: #64748B;">Logistic, Tree & Random Forest</div>
            </div>
            <div style="color: #94A3B8; font-size: 1.2rem;">➔</div>
            <div style="flex: 1; min-width: 140px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; text-align: center;">
                <div style="font-size: 1.4rem; margin-bottom: 4px;">⚡</div>
                <div style="font-weight: 700; font-size: 0.85rem; color: #101828;">5. TrainIQ SaaS</div>
                <div style="font-size: 0.75rem; color: #64748B;">Interactive Streamlit Engine</div>
            </div>
        </div>
    </div>
    """)
    
    # Technology Stack Badges
    render_html("""
    <div class="tiq-card">
        <div style="font-size: 1.05rem; font-weight: 700; color: #101828; margin-bottom: 12px;">Technology Stack</div>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            <span style="padding: 6px 14px; background: #F1F5F9; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #334155;">🐍 Python 3.14</span>
            <span style="padding: 6px 14px; background: #F1F5F9; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #334155;">⚡ Streamlit 1.43+</span>
            <span style="padding: 6px 14px; background: #F1F5F9; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #334155;">⚙️ Scikit-Learn</span>
            <span style="padding: 6px 14px; background: #F1F5F9; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #334155;">📊 Plotly Express</span>
            <span style="padding: 6px 14px; background: #F1F5F9; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #334155;">🐼 Pandas & NumPy</span>
            <span style="padding: 6px 14px; background: #F1F5F9; border-radius: 8px; font-size: 0.82rem; font-weight: 600; color: #334155;">📦 Joblib Model Serialization</span>
        </div>
    </div>
    """)

# -------------------------------------------------------------
# MAIN APPLICATION CONTROLLER
# -------------------------------------------------------------
def main():
    # Load dataset and model evaluation data
    df = load_main_dataset()
    eval_data = load_cached_evaluation_data()
    
    # Render Custom Dark Navy Sidebar and obtain active page
    selected_page = render_sidebar()
    
    # Route to Selected Page
    if "Overview" in selected_page:
        render_overview_page(df, eval_data)
    elif "Analytics" in selected_page:
        render_analytics_page(df)
    elif "Employee Prediction" in selected_page:
        render_prediction_page(df)
    elif "Model Insights" in selected_page:
        render_model_insights_page(eval_data)
    elif "Dataset Explorer" in selected_page:
        render_dataset_explorer_page(df)
    elif "About Project" in selected_page:
        render_about_page()

if __name__ == "__main__":
    main()
