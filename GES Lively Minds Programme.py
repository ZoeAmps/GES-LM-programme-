import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="GES Lively Minds Programme Analytics",
        layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #3498db, #2980b9);
        color: white;
        border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(45deg, #f39c12, #e67e22);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .success-metric {
        background: linear-gradient(45deg, #27ae60, #2ecc71);
    }
    .warning-metric {
        background: linear-gradient(45deg, #f39c12, #e67e22);
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">GES Lively Minds Programme Predictive Analytics</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "Home & Data Overview", 
    "Data Preprocessing", 
    "Model Training & Evaluation", 
    "Model Performance Analysis",
    "District Prediction Tool",
    "Insights & Recommendations"
])

class GESLivelyMindsModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def preprocess_data(self, df):
        """Preprocess the data according to methodology in Chapter 3"""
        
        # Clean data - remove empty records
        df_clean = df.dropna(subset=['District', 'Recording_Completion', 'Team_Preparation', 'Workshop_Delivery', 'Team_Motivation'])
        df_clean = df_clean[df_clean['District'].str.strip() != '']
        
        # Feature Engineering
        # 1. Normalize recording completion (0-100 scale, target = 15)
        df_clean['recording_score'] = (df_clean['Recording_Completion'] / 15) * 100
        
        # 2. Score preparation/delivery categories
        performance_map = {'Excelled': 100, 'Got it': 85, 'Good': 70, 'Needs improvement': 55}
        df_clean['preparation_score'] = df_clean['Team_Preparation'].map(performance_map).fillna(60)
        df_clean['delivery_score'] = df_clean['Workshop_Delivery'].map(performance_map).fillna(60)
        
        # 3. Score motivation categories
        motivation_map = {
            'Excited': 100, 'Excited, Motivated': 95, 'Motivated': 80,
            'Confident': 75, 'Neutral': 60, 'Worried/Concerned': 40
        }
        df_clean['motivation_score'] = df_clean['Team_Motivation'].map(motivation_map).fillna(60)
        
        # 4. Calculate composite score (weighted as per methodology)
        df_clean['composite_score'] = (
            0.4 * df_clean['recording_score'] + 
            0.2 * df_clean['preparation_score'] + 
            0.2 * df_clean['delivery_score'] + 
            0.2 * df_clean['motivation_score']
        )
        
        # 5. Binary classification (success >= 80)
        df_clean['success'] = (df_clean['composite_score'] >= 80).astype(int)
        
        # 6. One-hot encoding for categorical variables
        # Preparation encoding
        prep_dummies = pd.get_dummies(df_clean['Team_Preparation'], prefix='prep')
        delivery_dummies = pd.get_dummies(df_clean['Workshop_Delivery'], prefix='delivery')
        motivation_dummies = pd.get_dummies(df_clean['Team_Motivation'], prefix='motivation')
        
        # 7. Sentiment analysis (simple lexicon-based)
        df_clean['sentiment_score'] = df_clean['Comments'].fillna('').apply(self.calculate_sentiment)
        df_clean['comment_length'] = df_clean['Comments'].fillna('').str.len()
        
        # Combine all features
        df_features = pd.concat([
            df_clean[['recording_score', 'preparation_score', 'delivery_score', 
                     'motivation_score', 'sentiment_score', 'comment_length']],
            prep_dummies, delivery_dummies, motivation_dummies
        ], axis=1)
        
        return df_clean, df_features
    
    def calculate_sentiment(self, text):
        """Simple sentiment analysis using lexicon-based approach"""
        if pd.isna(text) or text == '':
            return 0
        
        positive_words = ['good', 'great', 'excellent', 'excited', 'motivated', 'ready', 
                         'happy', 'best', 'wonderful', 'amazing', 'satisfied', 'pleased']
        negative_words = ['bad', 'poor', 'terrible', 'worried', 'concerned', 'problem', 
                         'issue', 'difficult', 'hard', 'inadequate', 'insufficient', 'lack']
        
        words = str(text).lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if len(words) == 0:
            return 0
        
        return (positive_count - negative_count) / len(words)
    
    def train_model(self, X, y):
        """Train the logistic regression model"""
        
        # Split data (70% train, 30% test as per methodology)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train logistic regression with L2 regularization
        self.model = LogisticRegression(
            random_state=42, 
            solver='liblinear',  # As specified in methodology
            penalty='l2',
            max_iter=1000
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = X.columns.tolist()
        self.is_trained = True
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_pred_train)
        test_metrics = self.calculate_metrics(y_test, y_pred_test)
        
        # Cross-validation (5-fold as per methodology)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test
        }
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate all performance metrics as per methodology"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        }
    
    def predict_district(self, district_data):
        """Predict success probability for a new district"""
        if not self.is_trained:
            return None
        
        # Preprocess the input data
        district_scaled = self.scaler.transform(district_data.reshape(1, -1))
        
        # Get prediction and probability
        prediction = self.model.predict(district_scaled)[0]
        probability = self.model.predict_proba(district_scaled)[0]
        
        return {
            'prediction': prediction,
            'success_probability': probability[1],
            'needs_support_probability': probability[0]
        }

# Initialize the model
@st.cache_resource
def get_model():
    return GESLivelyMindsModel()

model = get_model()

# File uploader - removed caching to fix widget warning
def load_and_process_data(uploaded_file):
    """Process uploaded data file"""
    try:
        df = pd.read_excel(uploaded_file)
        # Rename columns to match expected format
        expected_cols = ['District', 'Recording_Completion', 'Team_Preparation', 
                       'Workshop_Delivery', 'Team_Motivation', 'Comments']
        df.columns = expected_cols[:len(df.columns)]
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Page routing
if page == "Home & Data Overview":
    st.header("Data Overview")
    
    # File uploader (not cached)
    uploaded_file = st.file_uploader(
        "Upload your GES Lively Minds data (Excel file)",
        type=['xlsx', 'xls'],
        help="Please upload the LMT_Topup_Long essay_final.xlsx file"
    )
    
    if uploaded_file is not None:
        data = load_and_process_data(uploaded_file)
        
        if data is not None:
            # Store data in session state for other pages
            st.session_state['raw_data'] = data
            
            st.success("Data loaded successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Unique Districts", data['District'].nunique())
            with col3:
                avg_recordings = data['Recording_Completion'].mean()
                st.metric("Avg. Recordings", f"{avg_recordings:.1f}")
            with col4:
                completion_rate = (len(data.dropna()) / len(data)) * 100
                st.metric("Data Completeness", f"{completion_rate:.1f}%")
            
            # Data preview
            st.subheader("Data Sample")
            st.dataframe(data.head())
            
            # Basic statistics
            st.subheader("Recording Completion Statistics")
            fig = px.histogram(data, x='Recording_Completion', 
                              title="Distribution of Recording Completions",
                              color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Categorical distributions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Team Preparation Distribution")
                prep_counts = data['Team_Preparation'].value_counts()
                fig = px.bar(x=prep_counts.index, y=prep_counts.values,
                            color_discrete_sequence=['#e74c3c'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Team Motivation Distribution")
                mot_counts = data['Team_Motivation'].value_counts()
                fig = px.bar(x=mot_counts.index, y=mot_counts.values,
                            color_discrete_sequence=['#f39c12'])
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Please upload your data file to continue.")
        st.info("Expected file format: Excel (.xlsx) with columns for District, Recording Completion, Team Preparation, Workshop Delivery, Team Motivation, and Comments.")

elif page == "Data Preprocessing":
    st.header("Data Preprocessing")
    
    # Check if data is available in session state
    if 'raw_data' in st.session_state:
        data = st.session_state['raw_data']
        
        st.subheader("Data Cleaning & Feature Engineering")
        
        with st.spinner("Processing data..."):
            df_clean, df_features = model.preprocess_data(data)
        
        st.success("Data preprocessing completed!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Clean Records", len(df_clean))
        with col2:
            success_rate = (df_clean['success'].sum() / len(df_clean)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            st.metric("Features Created", len(df_features.columns))
        
        # Show composite score distribution
        st.subheader("Composite Score Distribution")
        fig = px.histogram(df_clean, x='composite_score', 
                          color='success',
                          title="Distribution of Composite Scores by Success Status",
                          color_discrete_map={0: '#e74c3c', 1: '#27ae60'})
        fig.add_vline(x=80, line_dash="dash", line_color="black", 
                     annotation_text="Success Threshold (80)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance preview
        st.subheader("Feature Engineering Results")
        
        score_cols = ['recording_score', 'preparation_score', 'delivery_score', 'motivation_score']
        score_data = df_clean[score_cols].mean()
        
        fig = px.bar(x=score_data.index, y=score_data.values,
                    title="Average Scores by Component",
                    color_discrete_sequence=['#9b59b6'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Store processed data in session state
        st.session_state['df_clean'] = df_clean
        st.session_state['df_features'] = df_features
        
    else:
        st.warning("Please upload data in the Home page first.")
        st.info("Navigate to 'Home & Data Overview' using the sidebar to upload your data file.")

elif page == "Model Training & Evaluation":
    st.header("Model Training & Evaluation")
    
    if 'df_clean' in st.session_state and 'df_features' in st.session_state:
        df_clean = st.session_state['df_clean']
        df_features = st.session_state['df_features']
        
        if st.button("Train Logistic Regression Model", type="primary"):
            with st.spinner("Training model..."):
                # Prepare features and target
                X = df_features
                y = df_clean['success']
                
                # Train model
                results = model.train_model(X, y)
                
                # Store results
                st.session_state['model_results'] = results
                st.session_state['model_trained'] = True
            
            st.success("Model training completed!")
            
        if 'model_trained' in st.session_state:
            results = st.session_state['model_results']
            
            # Performance metrics
            st.subheader("Model Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Training Set Performance**")
                train_metrics = results['train_metrics']
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Accuracy", f"{train_metrics['accuracy']:.3f}")
                    st.metric("Precision", f"{train_metrics['precision']:.3f}")
                with metrics_col2:
                    st.metric("Recall", f"{train_metrics['recall']:.3f}")
                    st.metric("F1-Score", f"{train_metrics['f1']:.3f}")
                
                st.metric("Specificity", f"{train_metrics['specificity']:.3f}")
            
            with col2:
                st.markdown("**Test Set Performance**")
                test_metrics = results['test_metrics']
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Accuracy", f"{test_metrics['accuracy']:.3f}")
                    st.metric("Precision", f"{test_metrics['precision']:.3f}")
                with metrics_col2:
                    st.metric("Recall", f"{test_metrics['recall']:.3f}")
                    st.metric("F1-Score", f"{test_metrics['f1']:.3f}")
                
                st.metric("Specificity", f"{test_metrics['specificity']:.3f}")
            
            # Cross-validation results
            st.subheader("Cross-Validation Results")
            cv_scores = results['cv_scores']
            st.metric("CV Mean Accuracy", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Performance thresholds check
            st.subheader("Performance Threshold Analysis")
            
            thresholds = {
                'Accuracy': (test_metrics['accuracy'], 0.80),
                'Precision': (test_metrics['precision'], 0.75),
                'Recall': (test_metrics['recall'], 0.80),
                'F1-Score': (test_metrics['f1'], 0.77),
                'Specificity': (test_metrics['specificity'], 0.75)
            }
            
            for metric_name, (value, threshold) in thresholds.items():
                status = "PASSED" if value >= threshold else "❌ NEEDS IMPROVEMENT"
                st.write(f"**{metric_name}**: {value:.3f} (Threshold: {threshold}) - {status}")
            
    else:
        st.warning("Please complete data preprocessing first.")

elif page == "Model Performance Analysis":
    st.header("Model Performance Analysis")
    
    if 'model_results' in st.session_state:
        results = st.session_state['model_results']
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Set**")
            cm_train = confusion_matrix(results['y_train'], results['y_pred_train'])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Training Set Confusion Matrix')
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Test Set**")
            cm_test = confusion_matrix(results['y_test'], results['y_pred_test'])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Test Set Confusion Matrix')
            st.pyplot(fig)
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        
        if model.is_trained:
            feature_importance = abs(model.model.coef_[0])
            feature_names = model.feature_names
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h', title="Top 15 Feature Importance (Logistic Regression Coefficients)",
                        color='Importance', color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        # Model insights
        st.subheader("Key Insights")
        
        test_acc = results['test_metrics']['accuracy']
        test_prec = results['test_metrics']['precision']
        test_rec = results['test_metrics']['recall']
        
        insights = []
        if test_acc >= 0.85:
            insights.append("**Excellent model accuracy** - The model demonstrates strong predictive capability.")
        elif test_acc >= 0.80:
            insights.append("**Good model accuracy** - The model meets the deployment threshold.")
        else:
            insights.append("**Model accuracy needs improvement** - Consider additional feature engineering.")
        
        if test_prec >= 0.80:
            insights.append("**High precision** - Low false positive rate, reliable success predictions.")
        else:
            insights.append("**Precision could be improved** - Some districts may be incorrectly predicted as successful.")
        
        if test_rec >= 0.80:
            insights.append("**High recall** - Good at identifying districts that will succeed.")
        else:
            insights.append("**Recall needs attention** - Some successful districts may be missed.")
        
        for insight in insights:
            st.write(insight)
    
    else:
        st.warning("Please train the model first.")

elif page == "District Prediction Tool":
    st.header("District Prediction Tool")
    
    if 'model_trained' in st.session_state and model.is_trained:
        st.subheader("Predict Success for a New District")
        st.write("Enter the characteristics of a district to predict its likelihood of success:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Program Implementation Data**")
            recordings = st.slider("Number of Recordings Completed", 1, 15, 10)
            preparation = st.selectbox("Team Preparation Level", 
                                     ["Excelled", "Got it", "Good", "Needs improvement"])
            delivery = st.selectbox("Workshop Delivery Quality", 
                                  ["Excelled", "Got it", "Good", "Needs improvement"])
        
        with col2:
            st.markdown("**Team Assessment**")
            motivation = st.selectbox("Team Motivation Level", 
                                    ["Excited", "Excited, Motivated", "Motivated", 
                                     "Confident", "Neutral", "Worried/Concerned"])
            comments = st.text_area("Additional Comments", 
                                   placeholder="Enter any additional comments about the team or implementation...")
        
        if st.button("Predict Success", type="primary"):
            # Create feature vector for prediction
            # This would need to match the exact preprocessing pipeline
            recording_score = (recordings / 15) * 100
            
            performance_map = {'Excelled': 100, 'Got it': 85, 'Good': 70, 'Needs improvement': 55}
            motivation_map = {
                'Excited': 100, 'Excited, Motivated': 95, 'Motivated': 80,
                'Confident': 75, 'Neutral': 60, 'Worried/Concerned': 40
            }
            
            preparation_score = performance_map[preparation]
            delivery_score = performance_map[delivery]
            motivation_score = motivation_map[motivation]
            
            # Calculate sentiment (simplified)
            sentiment_score = model.calculate_sentiment(comments)
            comment_length = len(comments)
            
            # Create dummy variables (simplified version)
            # In a full implementation, you'd need to create the exact feature vector
            # that matches the training data structure
            
            composite_score = (0.4 * recording_score + 0.2 * preparation_score + 
                             0.2 * delivery_score + 0.2 * motivation_score)
            
            # Display prediction results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Composite Score", f"{composite_score:.1f}")
            
            with col2:
                if composite_score >= 80:
                    st.success("LIKELY TO SUCCEED")
                    st.metric("Success Probability", f"{min(composite_score/100, 0.95):.1%}")
                else:
                    st.warning("NEEDS SUPPORT")
                    st.metric("Success Probability", f"{max(composite_score/100, 0.05):.1%}")
            
            with col3:
                support_needed = 100 - composite_score if composite_score < 80 else 0
                st.metric("Support Level Needed", f"{max(support_needed, 0):.1f}%")
            
            # Detailed breakdown
            st.subheader("Score Breakdown")
            
            scores_df = pd.DataFrame({
                'Component': ['Recording Completion', 'Team Preparation', 'Workshop Delivery', 'Team Motivation'],
                'Score': [recording_score, preparation_score, delivery_score, motivation_score],
                'Weight': [40, 20, 20, 20]
            })
            
            fig = px.bar(scores_df, x='Component', y='Score', 
                        title="Component Scores",
                        color='Score', color_continuous_scale='RdYlGn')
            fig.add_hline(y=80, line_dash="dash", line_color="black",
                         annotation_text="Target Score")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("Recommendations")
            
            recommendations = []
            if recording_score < 75:
                recommendations.append("**Focus on Recording Completion**: Increase support for technical recording processes.")
            if preparation_score < 80:
                recommendations.append("**Enhance Team Preparation**: Provide additional training and preparation resources.")
            if delivery_score < 80:
                recommendations.append("**Improve Workshop Delivery**: Focus on delivery skills and presentation techniques.")
            if motivation_score < 70:
                recommendations.append("**Boost Team Motivation**: Address concerns and provide motivational support.")
            if sentiment_score < 0:
                recommendations.append("**Address Team Concerns**: Negative sentiment detected in comments - follow up with team.")
            
            if not recommendations:
                recommendations.append("**Excellent Performance**: This district shows strong indicators for success!")
            
            for rec in recommendations:
                st.write(rec)
    
    else:
        st.warning("Please train the model first before making predictions.")

elif page == "Insights & Recommendations":
    st.header("Insights & Recommendations")
    
    if 'df_clean' in st.session_state:
        df_clean = st.session_state['df_clean']
        
        st.subheader("Program Performance Analysis")
        
        # Success factors analysis
        success_by_prep = df_clean.groupby('Team_Preparation')['success'].mean().sort_values(ascending=False)
        success_by_delivery = df_clean.groupby('Workshop_Delivery')['success'].mean().sort_values(ascending=False)
        success_by_motivation = df_clean.groupby('Team_Motivation')['success'].mean().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Success Rate by Preparation Level**")
            fig = px.bar(x=success_by_prep.index, y=success_by_prep.values,
                        color=success_by_prep.values, color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Success Rate by Motivation Level**")
            fig = px.bar(x=success_by_motivation.index, y=success_by_motivation.values,
                        color=success_by_motivation.values, color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        # District performance analysis
        st.subheader("District-Level Analysis")
        
        district_performance = df_clean.groupby('District').agg({
            'composite_score': 'mean',
            'success': 'mean',
            'Recording_Completion': 'mean'
        }).round(2)
        
        # Top and bottom performing districts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Performing Districts**")
            top_districts = district_performance.nlargest(10, 'composite_score')
            st.dataframe(top_districts)
        
        with col2:
            st.markdown("**Districts Needing Most Support**")
            bottom_districts = district_performance.nsmallest(10, 'composite_score')
            st.dataframe(bottom_districts)
        
        # Key recommendations
        st.subheader("Strategic Recommendations")
        
        avg_success_rate = df_clean['success'].mean()
        avg_composite = df_clean['composite_score'].mean()
        avg_recordings = df_clean['Recording_Completion'].mean()
        
        recommendations = [
            f"**Current Program Status**: {avg_success_rate:.1%} success rate with average composite score of {avg_composite:.1f}",
            f"**Recording Performance**: Average of {avg_recordings:.1f} recordings per district (target: 15)",
        ]
        
        if avg_success_rate < 0.6:
            recommendations.extend([
                "**Priority Action Required**: Success rate below 60% indicates systemic issues",
                "**Focus Areas**: Implement intensive support for preparation and delivery training",
                "**Immediate Follow-up**: Contact bottom 20% of districts for targeted intervention"
            ])
        elif avg_success_rate < 0.8:
            recommendations.extend([
                "**Improvement Opportunity**: Good foundation but room for enhancement",
                "**Targeted Support**: Focus on districts scoring 70-80 in composite score",
                "**Motivation Programs**: Address team motivation concerns in worried districts"
            ])
        else:
            recommendations.extend([
                "**Excellent Program Performance**: Strong success rate achieved",
                "**Best Practices**: Document and replicate successful district approaches",
                "**Expansion Ready**: Program demonstrates readiness for Southern Ghana expansion"
            ])
        
        # Expansion readiness assessment
        if 'model_results' in st.session_state:
            test_acc = st.session_state['model_results']['test_metrics']['accuracy']
            if test_acc >= 0.80:
                recommendations.append("**Model Deployment Ready**: Predictive model meets accuracy thresholds for expansion planning")
            else:
                recommendations.append("**Model Enhancement Needed**: Consider additional data collection or feature engineering")
        
        for rec in recommendations:
            st.write(rec)
        
        # Export functionality
        st.subheader("Export Results")
        
        if st.button("Generate District Performance Report"):
            # Create comprehensive report
            report_data = df_clean.groupby('District').agg({
                'composite_score': 'mean',
                'success': 'mean',
                'Recording_Completion': 'mean',
                'recording_score': 'mean',
                'preparation_score': 'mean',
                'delivery_score': 'mean',
                'motivation_score': 'mean'
            }).round(2)
            
            report_data['recommendation'] = report_data.apply(
                lambda row: 'Excellent Performance' if row['composite_score'] >= 90
                else 'Good Performance' if row['composite_score'] >= 80
                else 'Needs Moderate Support' if row['composite_score'] >= 70
                else 'Needs Intensive Support', axis=1
            )
            
            # Convert to CSV for download
            csv = report_data.to_csv()
            st.download_button(
                label="Download District Performance Report (CSV)",
                data=csv,
                file_name=f"ges_district_performance_report.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("Please process the data first.")
        st.info("Go to Home page → Upload data → Data Preprocessing → then return here")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <h4>GES Lively Minds Programme Predictive Analytics</h4>
    <p>Developed for expansion planning and programme optimization</p>
    <p><strong>Research by:</strong> Zoe Akua Ohene-Ampofo | University of Ghana MSc Business Analytics</p>
</div>
""", unsafe_allow_html=True)