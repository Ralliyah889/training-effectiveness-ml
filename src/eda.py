import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_target_distribution(df):
    """Plots the distribution of the Outcome variable."""
    # Temporarily calculate outcome for EDA
    temp_df = df.copy()
    temp_df['Outcome'] = temp_df['PerformanceRating'].apply(lambda x: 'Successful' if x == 4 else 'Needs Improvement')
    val_counts = temp_df['Outcome'].value_counts().reset_index()
    val_counts.columns = ['Outcome', 'Count']
    fig = px.pie(val_counts, values='Count', names='Outcome', title='Distribution of Post-Training Outcomes', color_discrete_sequence=['#ef553b', '#636efa'])
    return fig

def plot_training_vs_outcome(df):
    temp_df = df.copy()
    temp_df['Outcome'] = temp_df['PerformanceRating'].apply(lambda x: 'Successful' if x == 4 else 'Needs Improvement')
    fig = px.histogram(temp_df, x='TrainingTimesLastYear', color='Outcome', barmode='group',
                       title='Training Sessions Last Year vs Outcome',
                       labels={'TrainingTimesLastYear': 'Training Sessions'})
    return fig

def plot_department_vs_outcome(df):
    temp_df = df.copy()
    temp_df['Outcome'] = temp_df['PerformanceRating'].apply(lambda x: 'Successful' if x == 4 else 'Needs Improvement')
    fig = px.histogram(temp_df, x='Department', color='Outcome', barmode='group',
                       title='Department vs Outcome')
    return fig

def plot_experience_vs_outcome(df):
    temp_df = df.copy()
    temp_df['Outcome'] = temp_df['PerformanceRating'].apply(lambda x: 'Successful' if x == 4 else 'Needs Improvement')
    fig = px.box(temp_df, x='Outcome', y='TotalWorkingYears', color='Outcome',
                 title='Total Working Years vs Outcome')
    return fig

def plot_satisfaction_vs_outcome(df):
    temp_df = df.copy()
    temp_df['Outcome'] = temp_df['PerformanceRating'].apply(lambda x: 'Successful' if x == 4 else 'Needs Improvement')
    fig = px.histogram(temp_df, x='JobSatisfaction', color='Outcome', barmode='group',
                       title='Job Satisfaction vs Outcome')
    return fig

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=False, aspect="auto", title="Correlation Heatmap for Numerical Features", color_continuous_scale='RdBu_r')
    return fig
