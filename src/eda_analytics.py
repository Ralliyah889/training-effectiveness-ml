import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from theme import PLOTLY_LAYOUT_TEMPLATE, COLOR_PALETTE

def get_prepared_eda_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares dataframe with clear human-readable Outcome labels."""
    temp = df.copy()
    if 'PerformanceRating' in temp.columns and 'Outcome' not in temp.columns:
        temp['Outcome'] = temp['PerformanceRating'].apply(
            lambda x: 'Successful' if x == 4 else 'Needs Improvement'
        )
    elif 'Outcome' in temp.columns:
        temp['Outcome'] = temp['Outcome'].apply(
            lambda x: 'Successful' if x in [1, '1', 'Successful'] else 'Needs Improvement'
        )
    return temp

def create_target_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart for outcome distribution with clean SaaS styling."""
    data = get_prepared_eda_data(df)
    counts = data['Outcome'].value_counts().reset_index()
    counts.columns = ['Outcome', 'Count']
    total = counts['Count'].sum()
    counts['Percentage'] = (counts['Count'] / total * 100).round(1)
    
    colors = [COLOR_PALETTE['outcome_colors'].get(x, '#3563E9') for x in counts['Outcome']]
    
    fig = go.Figure(data=[go.Pie(
        labels=counts['Outcome'],
        values=counts['Count'],
        hole=0.62,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+percent',
        hovertemplate="<b>%{label}</b><br>Employees: %{value:,}<br>Share: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        **PLOTLY_LAYOUT_TEMPLATE,
        title=dict(text="<b>Post-Training Outcome Distribution</b>", font=dict(size=14, color="#101828")),
        annotations=[dict(
            text=f"<b>{total:,}</b><br><span style='font-size:10px; color:#64748B;'>Total</span>",
            x=0.5, y=0.5, font_size=16, font_family="Plus Jakarta Sans, sans-serif",
            showarrow=False
        )],
        showlegend=True,
        height=320
    )
    return fig

def create_training_vs_outcome_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart for training sessions attended last year vs outcome."""
    data = get_prepared_eda_data(df)
    grouped = data.groupby(['TrainingTimesLastYear', 'Outcome']).size().reset_index(name='Count')
    
    fig = px.bar(
        grouped,
        x='TrainingTimesLastYear',
        y='Count',
        color='Outcome',
        barmode='group',
        color_discrete_map=COLOR_PALETTE['outcome_colors'],
        labels={'TrainingTimesLastYear': 'Training Sessions Last Year', 'Count': 'Number of Employees'}
    )
    
    fig.update_layout(
        **PLOTLY_LAYOUT_TEMPLATE,
        title=dict(text="<b>Training Frequency vs Post-Training Outcome</b>", font=dict(size=14, color="#101828")),
        height=320,
        bargap=0.2,
        bargroupgap=0.1
    )
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        hovertemplate="<b>%{x} Sessions</b><br>Outcome: %{fullData.name}<br>Employees: %{y}<extra></extra>"
    )
    return fig

def create_department_outcome_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal stacked proportion chart by department."""
    data = get_prepared_eda_data(df)
    dept_group = data.groupby(['Department', 'Outcome']).size().unstack(fill_value=0)
    dept_pct = dept_group.div(dept_group.sum(axis=1), axis=0) * 100
    dept_pct = dept_pct.reset_index()
    
    fig = go.Figure()
    for col in ['Needs Improvement', 'Successful']:
        if col in dept_pct.columns:
            fig.add_trace(go.Bar(
                y=dept_pct['Department'],
                x=dept_pct[col],
                name=col,
                orientation='h',
                marker=dict(color=COLOR_PALETTE['outcome_colors'].get(col, '#3563E9')),
                hovertemplate=f"<b>%{{y}}</b><br>{col}: %{{x:.1f}}%<extra></extra>"
            ))
            
    fig.update_layout(
        **PLOTLY_LAYOUT_TEMPLATE,
        barmode='stack',
        title=dict(text="<b>Department-wise Outcome Proportion (%)</b>", font=dict(size=14, color="#101828")),
        height=300
    )
    fig.update_xaxes(title="Percentage (%)", range=[0, 100])
    fig.update_yaxes(title="")
    return fig

def create_experience_outcome_chart(df: pd.DataFrame) -> go.Figure:
    """Box plot of total working years across outcomes."""
    data = get_prepared_eda_data(df)
    fig = px.box(
        data,
        x='Outcome',
        y='TotalWorkingYears',
        color='Outcome',
        color_discrete_map=COLOR_PALETTE['outcome_colors'],
        labels={'TotalWorkingYears': 'Total Working Years', 'Outcome': 'Outcome'}
    )
    fig.update_layout(
        **PLOTLY_LAYOUT_TEMPLATE,
        title=dict(text="<b>Working Experience Distribution by Outcome</b>", font=dict(size=14, color="#101828")),
        height=320,
        showlegend=False
    )
    return fig

def create_satisfaction_outcome_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped chart for Job Satisfaction vs Outcome."""
    data = get_prepared_eda_data(df)
    grouped = data.groupby(['JobSatisfaction', 'Outcome']).size().reset_index(name='Count')
    
    fig = px.bar(
        grouped,
        x='JobSatisfaction',
        y='Count',
        color='Outcome',
        barmode='group',
        color_discrete_map=COLOR_PALETTE['outcome_colors'],
        labels={'JobSatisfaction': 'Job Satisfaction Rating (1-4)', 'Count': 'Employees'}
    )
    fig.update_layout(
        **PLOTLY_LAYOUT_TEMPLATE,
        title=dict(text="<b>Job Satisfaction vs Post-Training Outcome</b>", font=dict(size=14, color="#101828")),
        height=320
    )
    return fig

def create_job_role_outcome_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart for outcome count by job role."""
    data = get_prepared_eda_data(df)
    grouped = data.groupby(['JobRole', 'Outcome']).size().reset_index(name='Count')
    
    fig = px.bar(
        grouped,
        y='JobRole',
        x='Count',
        color='Outcome',
        barmode='group',
        orientation='h',
        color_discrete_map=COLOR_PALETTE['outcome_colors'],
        labels={'JobRole': '', 'Count': 'Employees'}
    )
    fig.update_layout(
        **PLOTLY_LAYOUT_TEMPLATE,
        title=dict(text="<b>Job Role Distribution across Outcomes</b>", font=dict(size=14, color="#101828")),
        height=380
    )
    return fig

def create_correlation_heatmap(df: pd.DataFrame, selected_cols: list = None) -> go.Figure:
    """Correlation heatmap with sleek Diverging palette."""
    numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
    
    # Drop zero variance or index cols
    drop = ['EmployeeCount', 'EmployeeNumber', 'StandardHours']
    numeric_df = numeric_df.drop(columns=[c for c in drop if c in numeric_df.columns])
    
    if selected_cols:
        valid_cols = [c for c in selected_cols if c in numeric_df.columns]
        if len(valid_cols) >= 2:
            numeric_df = numeric_df[valid_cols]
    else:
        # Pick top 10 relevant numerical columns for readability
        key_cols = ['Age', 'TotalWorkingYears', 'YearsAtCompany', 'TrainingTimesLastYear', 
                    'JobSatisfaction', 'EnvironmentSatisfaction', 'JobInvolvement', 
                    'MonthlyIncome', 'WorkLifeBalance', 'PerformanceRating']
        valid_cols = [c for c in key_cols if c in numeric_df.columns]
        if len(valid_cols) >= 2:
            numeric_df = numeric_df[valid_cols]
            
    corr = numeric_df.corr().round(2)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[[0, '#3563E9'], [0.5, '#FFFFFF'], [1, '#20A779']],
        zmin=-1,
        zmax=1
    )
    fig.update_layout(
        **PLOTLY_LAYOUT_TEMPLATE,
        title=dict(text="<b>Numerical Feature Correlation Matrix</b>", font=dict(size=14, color="#101828")),
        height=450
    )
    return fig

def create_confusion_matrix_chart(cm: np.ndarray, model_name: str) -> go.Figure:
    """Interactive heatmap for confusion matrix with annotations."""
    labels = ['Needs Improvement (0)', 'Successful (1)']
    
    # Text matrix
    text_matrix = [[f"<b>{val:,}</b>" for val in row] for row in cm]
    
    fig = px.imshow(
        cm,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        color_continuous_scale=[[0, '#F8FAFC'], [0.2, '#EDF4FF'], [1, '#3563E9']],
        text_auto=False
    )
    
    # Custom annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig.add_annotation(
                x=j, y=i,
                text=text_matrix[i][j],
                showarrow=False,
                font=dict(color="#101828" if cm[i][j] < (cm.max()/2) else "#FFFFFF", size=15, family="Plus Jakarta Sans, sans-serif")
            )
            
    fig.update_layout(
        **PLOTLY_LAYOUT_TEMPLATE,
        title=dict(text=f"<b>Confusion Matrix — {model_name}</b> (Test Set: 294 Samples)", font=dict(size=14, color="#101828")),
        height=320
    )
    fig.update_xaxes(title="Model Prediction", side="bottom")
    fig.update_yaxes(title="True Outcome")
    return fig

def create_feature_importance_chart(importances: np.ndarray, feature_names: list, top_n: int = 10) -> go.Figure:
    """Clean bar chart for feature importances."""
    fi_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(top_n)
    
    fig = px.bar(
        fi_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=[[0, '#93C5FD'], [1, '#3563E9']],
        labels={'Importance': 'Relative Importance', 'Feature': ''}
    )
    fig.update_layout(
        **PLOTLY_LAYOUT_TEMPLATE,
        title=dict(text=f"<b>Top {top_n} Influential Features (Random Forest)</b>", font=dict(size=14, color="#101828")),
        height=360,
        coloraxis_showscale=False
    )
    return fig
