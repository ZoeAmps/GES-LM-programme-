import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="GES Lively Minds Programme Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e5c8a;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        color: #155724;
    }
    .warning-metric {
        background-color: #fff3cd;
        color: #856404;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}

# Data preprocessing functions
def clean_data(df):
    """Clean and preprocess the uploaded data"""
    
    # Column mappings based on actual dataset structure
    column_mapping = {
        'District': 'district',
        'How many recordings did your District produce successfully?': 'recordings',
        "Rate your team's preparation for the workshop": 'team_preparation',
        "Rate your team's delivery of the workshop": 'workshop_delivery',
        "How is your team feeling about this upcoming season?": 'team_motivation',
        "Comments on team feeling": 'comments_feeling',
        "Any other highlights, challenges or comments?": 'comments_other',
        'Submit Date': 'submit_date'
    }
    
    # Select and rename columns
    available_cols = [col for col in column_mapping.keys() if col in df.columns]
    df_clean = df[available_cols].copy()
    df_clean.rename(columns={col: column_mapping[col] for col in available_cols}, inplace=True)
    
    # Clean recordings column
    df_clean['recordings'] = pd.to_numeric(df_clean['recordings'], errors='coerce')
    
    # Remove rows with missing critical data
    df_clean = df_clean.dropna(subset=['district', 'recordings', 'team_preparation', 
                                     'workshop_delivery', 'team_motivation'])
    
    # Combine comments
    df_clean['combined_comments'] = (
        df_clean.get('comments_feeling', '').fillna('').astype(str) + ' ' + 
        df_clean.get('comments_other', '').fillna('').astype(str)
    ).str.strip()
    
    # Clean text data
    for col in ['team_preparation', 'workshop_delivery', 'team_motivation']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].str.strip()
    
    return df_clean

def create_composite_score(df):
    """Create composite performance score"""
    
    # Recording Score (max 12 based on data analysis)
    df['recording_score'] = (df['recordings'] / 12) * 100
    df['recording_score'] = np.clip(df['recording_score'], 0, 100)
    
    # Preparation Score
    prep_mapping = {
        'Poor': 20, 'Fair': 40, 'Good': 60, 'Very Good': 80, 'Excellent': 100
    }
    df['preparation_score'] = df['team_preparation'].map(prep_mapping).fillna(0)
    
    # Delivery Score
    delivery_mapping = {
        'Poor': 20, 'Fair': 40, 'Good': 60, 'Very Good': 80, 'Excellent': 100
    }
    df['delivery_score'] = df['workshop_delivery'].map(delivery_mapping).fillna(0)
    
    # Motivation Score
    motivation_mapping = {
        'Worried/Concerned': 20, 'Neutral': 40, 'Motivated': 60, 
        'Excited': 80, 'Excited, Motivated': 100
    }
    df['motivation_score'] = df['team_motivation'].map(motivation_mapping).fillna(0)
    
    # Composite Score with weights from research
    df['composite_score'] = (
        0.4 * df['recording_score'] + 
        0.2 * df['preparation_score'] + 
        0.2 * df['delivery_score'] + 
        0.2 * df['motivation_score']
    )
    
    # Success Classification
    df['success'] = (df['composite_score'] >= 80).astype(int)
    
    return df

def perform_sentiment_analysis(df):
    """Perform sentiment analysis on comments"""
    
    def get_sentiment(text):
        if pd.isna(text) or text.strip() == '':
            return 0, 'neutral'
        
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            category = 'positive'
        elif polarity < -0.1:
            category = 'negative'
        else:
            category = 'neutral'
            
        return polarity, category
    
    df[['sentiment_polarity', 'sentiment_category']] = df['combined_comments'].apply(
        lambda x: pd.Series(get_sentiment(x))
    )
    
    return df

def prepare_features(df):
    """Prepare features for machine learning"""
    
    # One-hot encoding for categorical variables
    categorical_features = ['team_preparation', 'workshop_delivery', 'team_motivation']
    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
    
    # Select features for modeling
    feature_cols = [col for col in df_encoded.columns if any(cat in col for cat in categorical_features)]
    feature_cols.extend(['recording_score', 'preparation_score', 'delivery_score', 'motivation_score', 'sentiment_polarity'])
    
    # Remove columns that don't exist
    feature_cols = [col for col in feature_cols if col in df_encoded.columns]
    
    X = df_encoded[feature_cols]
    y = df_encoded['success']
    
    return X, y, feature_cols

def train_models(X, y):
    """Train multiple classification models"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        if name == 'SVM':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'specificity': recall_score(y_test, y_pred, pos_label=0),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        results[name]['cv_mean'] = cv_scores.mean()
        results[name]['cv_std'] = cv_scores.std()
    
    return results, X_test, y_test, scaler

# Visualization functions
def plot_performance_comparison(results):
    """Plot model performance comparison"""
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'specificity']
    model_names = list(results.keys())
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [results[model][metric] for model in model_names]
        fig.add_trace(go.Scatter(
            x=model_names,
            y=values,
            mode='lines+markers',
            name=metric.replace('_', ' ').title(),
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    # Add threshold lines
    thresholds = {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.8, 
                 'f1_score': 0.77, 'specificity': 0.75}
    
    for metric, threshold in thresholds.items():
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"{metric.title()} Threshold: {threshold}")
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Performance Score",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_confusion_matrices(results):
    """Plot confusion matrices for all models"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(results.keys()),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for i, (name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        row, col = positions[i]
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Needs Support', 'Success'],
                y=['Needs Support', 'Success'],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
                colorscale='Blues',
                showscale=i == 0
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title="Confusion Matrices Comparison",
        height=600
    )
    
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance"""
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Feature Importance - {model_name}',
        labels={'importance': 'Importance Score', 'feature': 'Features'}
    )
    
    fig.update_layout(height=max(400, len(feature_names) * 25))
    
    return fig

def plot_district_performance_scatter(df):
    """Create scatter plot of district performance"""
    
    # Transform sentiment polarity to positive values for size parameter
    # Original range: -1 to 1, New range: 0 to 2
    df_plot = df.copy()
    df_plot['sentiment_size'] = df_plot['sentiment_polarity'] + 1
    
    # Ensure minimum size for visibility
    df_plot['sentiment_size'] = np.maximum(df_plot['sentiment_size'], 0.1)
    
    fig = px.scatter(
        df_plot,
        x='composite_score',
        y='recordings',
        color='success',
        size='sentiment_size',
        hover_data=['district', 'team_preparation', 'workshop_delivery', 'team_motivation', 'sentiment_polarity'],
        color_discrete_map={0: 'red', 1: 'green'},
        title="District Performance Analysis",
        labels={
            'composite_score': 'Composite Performance Score',
            'recordings': 'Number of Recordings',
            'success': 'Classification',
            'sentiment_size': 'Sentiment (transformed for size)'
        }
    )
    
    # Add success threshold line
    fig.add_vline(x=80, line_dash="dash", line_color="black", 
                 annotation_text="Success Threshold (80)")
    
    fig.update_layout(height=500)
    
    return fig

def plot_sentiment_distribution(df):
    """Plot sentiment analysis results"""
    
    sentiment_counts = df['sentiment_category'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution in Feedback Comments",
        color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    )
    
    return fig

# Export functions
def create_download_link(df, filename, file_format='csv'):
    """Create download link for dataframe"""
    
    if file_format == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}.csv</a>'
    elif file_format == 'excel':
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:file/xlsx;base64,{b64}" download="{filename}.xlsx">Download {filename}.xlsx</a>'
    
    return href

# Main application
def main():
    init_session_state()
    
    st.markdown('<div class="main-header">GES Lively Minds Programme Analysis Platform</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    This platform analyzes the effectiveness of the GES Lively Minds Radio Top-Up Programme 
    and creates predictive models for Southern Ghana expansion planning.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Analysis Page", [
        "Data Upload & Overview",
        "Data Preprocessing",
        "Exploratory Analysis", 
        "Model Training & Evaluation",
        "Sentiment Analysis",
        "Predictions & Export"
    ])
    
    # Page routing
    if page == "Data Upload & Overview":
        data_upload_page()
    elif page == "Data Preprocessing":
        preprocessing_page()
    elif page == "Exploratory Analysis":
        exploratory_analysis_page()
    elif page == "Model Training & Evaluation":
        model_training_page()
    elif page == "Sentiment Analysis":
        sentiment_analysis_page()
    elif page == "Predictions & Export":
        predictions_page()

def data_upload_page():
    """Data upload and overview page"""
    
    st.markdown('<div class="sub-header">Data Upload & Overview</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload your Northern Ghana Top-up Reports dataset",
        type=['xlsx', 'csv'],
        help="Upload the Excel or CSV file containing the programme data"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            st.session_state.data = df
            st.success(f"Data uploaded successfully! Shape: {df.shape}")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Show sample data
            st.subheader("Sample Data Preview")
            st.dataframe(df.head(10))
            
            # Column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(col_info)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        st.info("Please upload a dataset to begin analysis.")

def preprocessing_page():
    """Data preprocessing page"""
    
    st.markdown('<div class="sub-header">Data Preprocessing</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload data first.")
        return
    
    df = st.session_state.data.copy()
    
    st.write("**Original Data Shape:**", df.shape)
    
    # Clean data
    with st.spinner("Cleaning and preprocessing data..."):
        df_clean = clean_data(df)
        df_processed = create_composite_score(df_clean)
        df_processed = perform_sentiment_analysis(df_processed)
    
    st.session_state.processed_data = df_processed
    st.success(f"Data preprocessing completed! New shape: {df_processed.shape}")
    
    # Display preprocessing results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Quality Summary")
        st.write("**Records retained:**", len(df_processed))
        st.write("**Records removed:**", len(df) - len(df_processed))
        st.write("**Completion rate:**", f"{len(df_processed)/len(df)*100:.1f}%")
        
        # Success distribution
        success_dist = df_processed['success'].value_counts()
        st.write("**Success Classification:**")
        st.write(f"- Success (≥80): {success_dist.get(1, 0)} districts")
        st.write(f"- Needs Support (<80): {success_dist.get(0, 0)} districts")
    
    with col2:
        st.subheader("Score Distributions")
        
        fig = make_subplots(rows=2, cols=2, 
                          subplot_titles=['Recording Score', 'Preparation Score', 
                                        'Delivery Score', 'Motivation Score'])
        
        scores = ['recording_score', 'preparation_score', 'delivery_score', 'motivation_score']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for score, (row, col) in zip(scores, positions):
            fig.add_trace(
                go.Histogram(x=df_processed[score], name=score.replace('_', ' ').title()),
                row=row, col=col
            )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show processed data sample
    st.subheader("Processed Data Sample")
    display_cols = ['district', 'recordings', 'team_preparation', 'workshop_delivery', 
                   'team_motivation', 'composite_score', 'success']
    available_cols = [col for col in display_cols if col in df_processed.columns]
    st.dataframe(df_processed[available_cols].head(10))

def exploratory_analysis_page():
    """Exploratory data analysis page"""
    
    st.markdown('<div class="sub-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("Please complete data preprocessing first.")
        return
    
    df = st.session_state.processed_data
    
    # Performance overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Composite Score", f"{df['composite_score'].mean():.1f}")
    with col2:
        st.metric("Success Rate", f"{df['success'].mean()*100:.1f}%")
    with col3:
        st.metric("Mean Recordings", f"{df['recordings'].mean():.1f}")
    with col4:
        st.metric("Total Districts", df['district'].nunique())
    
    # District performance scatter plot
    st.subheader("District Performance Analysis")
    scatter_fig = plot_district_performance_scatter(df)
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Performance by categorical variables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Success Rate by Team Preparation")
        prep_success = df.groupby('team_preparation')['success'].mean().reset_index()
        prep_success = prep_success.sort_values('success', ascending=True)
        
        fig = px.bar(prep_success, x='success', y='team_preparation', 
                    orientation='h', title="Success Rate by Preparation Level")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Success Rate by Team Motivation")
        motiv_success = df.groupby('team_motivation')['success'].mean().reset_index()
        motiv_success = motiv_success.sort_values('success', ascending=True)
        
        fig = px.bar(motiv_success, x='success', y='team_motivation',
                    orientation='h', title="Success Rate by Motivation Level")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Score Correlations")
    corr_cols = ['recording_score', 'preparation_score', 'delivery_score', 
                'motivation_score', 'composite_score']
    available_corr_cols = [col for col in corr_cols if col in df.columns]
    
    if len(available_corr_cols) > 1:
        corr_matrix = df[available_corr_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(title="Score Correlation Matrix", height=400)
        st.plotly_chart(fig, use_container_width=True)

def model_training_page():
    """Model training and evaluation page"""
    
    st.markdown('<div class="sub-header">Model Training & Evaluation</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("Please complete data preprocessing first.")
        return
    
    df = st.session_state.processed_data
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models..."):
            X, y, feature_names = prepare_features(df)
            results, X_test, y_test, scaler = train_models(X, y)
            
            st.session_state.models = results
            st.session_state.feature_names = feature_names
            st.session_state.scaler = scaler
        
        st.success("Models trained successfully!")
    
    if st.session_state.models:
        results = st.session_state.models
        
        # Performance summary table
        st.subheader("Model Performance Summary")
        
        performance_data = []
        thresholds = {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.8, 
                     'f1_score': 0.77, 'specificity': 0.75}
        
        for model_name, result in results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.3f}",
                'Precision': f"{result['precision']:.3f}",
                'Recall': f"{result['recall']:.3f}",
                'F1-Score': f"{result['f1_score']:.3f}",
                'Specificity': f"{result['specificity']:.3f}",
                'ROC-AUC': f"{result['roc_auc']:.3f}",
                'CV Mean': f"{result['cv_mean']:.3f}",
                'Meets Thresholds': 'Yes' if all(
                    result[metric] >= threshold for metric, threshold in thresholds.items()
                ) else 'No'
            }
            performance_data.append(row)
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Performance comparison chart
        st.subheader("Performance Comparison")
        comparison_fig = plot_performance_comparison(results)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        cm_fig = plot_confusion_matrices(results)
        st.plotly_chart(cm_fig, use_container_width=True)
        
        # Feature importance for best model
        best_model = max(results.keys(), key=lambda k: results[k]['f1_score'])
        st.subheader(f"Feature Importance - Best Model ({best_model})")
        
        importance_fig = plot_feature_importance(
            results[best_model]['model'], 
            st.session_state.feature_names, 
            best_model
        )
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
        
        # Model recommendations
        st.subheader("Model Recommendations")
        st.markdown(f"""
        **Best Performing Model:** {best_model}
        
        **Key Insights:**
        - F1-Score: {results[best_model]['f1_score']:.3f}
        - Meets all performance thresholds: {'Yes' if performance_df.loc[performance_df['Model'] == best_model, 'Meets Thresholds'].iloc[0] == 'Yes' else 'No'}
        - Cross-validation stability: {results[best_model]['cv_std']:.3f} (lower is better)
        """)

def sentiment_analysis_page():
    """Sentiment analysis page"""
    
    st.markdown('<div class="sub-header">Sentiment Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("Please complete data preprocessing first.")
        return
    
    df = st.session_state.processed_data
    
    # Sentiment overview
    col1, col2, col3 = st.columns(3)
    
    sentiment_counts = df['sentiment_category'].value_counts()
    
    with col1:
        st.metric("Positive Comments", sentiment_counts.get('positive', 0))
    with col2:
        st.metric("Neutral Comments", sentiment_counts.get('neutral', 0))
    with col3:
        st.metric("Negative Comments", sentiment_counts.get('negative', 0))
    
    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_fig = plot_sentiment_distribution(df)
    st.plotly_chart(sentiment_fig, use_container_width=True)
    
    # Sentiment vs Performance
    st.subheader("Sentiment vs Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment by success
        sentiment_success = pd.crosstab(df['sentiment_category'], df['success'], normalize='columns')
        
        fig = px.bar(
            sentiment_success.T,
            title="Sentiment Distribution by Success Classification",
            labels={'value': 'Proportion', 'index': 'Success Classification'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average sentiment by composite score
        fig = px.scatter(
            df,
            x='composite_score',
            y='sentiment_polarity', 
            color='sentiment_category',
            title="Sentiment Polarity vs Composite Score",
            labels={
                'composite_score': 'Composite Score',
                'sentiment_polarity': 'Sentiment Polarity'
            }
        )
        fig.add_vline(x=80, line_dash="dash", annotation_text="Success Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample comments by sentiment
    st.subheader("Sample Comments by Sentiment Category")
    
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment in sentiment_counts.index:
            st.write(f"**{sentiment.title()} Comments:**")
            sample_comments = df[df['sentiment_category'] == sentiment]['combined_comments'].dropna()
            if len(sample_comments) > 0:
                for i, comment in enumerate(sample_comments.head(3)):
                    if comment.strip():
                        st.write(f"{i+1}. {comment[:200]}...")
            st.write("")

def predictions_page():
    """Predictions and export page"""
    
    st.markdown('<div class="sub-header">Predictions & Export</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("Please complete data preprocessing first.")
        return
    
    df = st.session_state.processed_data
    
    # Single district prediction
    st.subheader("Single District Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        recordings = st.slider("Number of Recordings", 1, 12, 8)
        preparation = st.selectbox("Team Preparation", 
                                 ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        delivery = st.selectbox("Workshop Delivery",
                              ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    
    with col2:
        motivation = st.selectbox("Team Motivation",
                                ['Worried/Concerned', 'Neutral', 'Motivated', 'Excited', 'Excited, Motivated'])
        comments = st.text_area("Comments (optional)", "")
    
    if st.button("Predict District Success"):
        # Calculate scores
        recording_score = (recordings / 12) * 100
        
        prep_mapping = {'Poor': 20, 'Fair': 40, 'Good': 60, 'Very Good': 80, 'Excellent': 100}
        preparation_score = prep_mapping[preparation]
        delivery_score = prep_mapping[delivery]
        
        motiv_mapping = {'Worried/Concerned': 20, 'Neutral': 40, 'Motivated': 60, 
                        'Excited': 80, 'Excited, Motivated': 100}
        motivation_score = motiv_mapping[motivation]
        
        composite_score = (0.4 * recording_score + 0.2 * preparation_score + 
                          0.2 * delivery_score + 0.2 * motivation_score)
        
        success = "Success" if composite_score >= 80 else "Needs Support"
        
        # Sentiment
        if comments:
            blob = TextBlob(comments)
            sentiment = blob.sentiment.polarity
        else:
            sentiment = 0
        
        # Display prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Composite Score", f"{composite_score:.1f}")
        with col2:
            color = "green" if success == "Success" else "orange"
            st.markdown(f'<div style="color: {color}; font-weight: bold; font-size: 20px;">{success}</div>', 
                       unsafe_allow_html=True)
        with col3:
            st.metric("Sentiment", f"{sentiment:.2f}")
    
    # Export options
    st.subheader("Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Processed Data"):
            csv_link = create_download_link(df, "processed_data", "csv")
            st.markdown(csv_link, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.models and st.button("Export Model Results"):
            results_data = []
            for model_name, result in st.session_state.models.items():
                results_data.append({
                    'Model': model_name,
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1_Score': result['f1_score'],
                    'Specificity': result['specificity'],
                    'ROC_AUC': result['roc_auc'],
                    'CV_Mean': result['cv_mean'],
                    'CV_Std': result['cv_std']
                })
            results_df = pd.DataFrame(results_data)
            csv_link = create_download_link(results_df, "model_results", "csv")
            st.markdown(csv_link, unsafe_allow_html=True)
    
    with col3:
        # District summary
        if st.button("Export District Summary"):
            district_summary = df.groupby('district').agg({
                'composite_score': 'mean',
                'success': 'mean',
                'recordings': 'mean',
                'sentiment_polarity': 'mean'
            }).reset_index()
            district_summary.columns = ['District', 'Avg_Composite_Score', 'Success_Rate', 
                                      'Avg_Recordings', 'Avg_Sentiment']
            csv_link = create_download_link(district_summary, "district_summary", "csv")
            st.markdown(csv_link, unsafe_allow_html=True)
    
    # Southern Ghana expansion recommendations
    st.subheader("Southern Ghana Expansion Recommendations")
    
    if st.session_state.models:
        best_model = max(st.session_state.models.keys(), 
                        key=lambda k: st.session_state.models[k]['f1_score'])
        
        st.markdown(f"""
        **Recommended Model for Southern Ghana:** {best_model}
        
        **Key Success Factors (based on model analysis):**
        1. **Recording Completion** (40% weight): Target ≥8 recordings per district
        2. **Team Preparation** (20% weight): Ensure "Good" or higher preparation levels
        3. **Workshop Delivery** (20% weight): Focus on delivery quality training
        4. **Team Motivation** (20% weight): Address concerns early to maintain motivation
        
        **Risk Indicators to Monitor:**
        - Districts with <6 recordings typically need additional support
        - "Worried/Concerned" motivation requires immediate intervention
        - Negative sentiment in feedback indicates implementation challenges
        
        **Expansion Strategy:**
        - Start with districts showing high baseline capacity
        - Implement early warning systems for motivation and sentiment
        - Ensure adequate regional support infrastructure
        """)

if __name__ == "__main__":
    main()
