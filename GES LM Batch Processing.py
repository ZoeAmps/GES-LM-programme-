#!/usr/bin/env python3
"""
GES Lively Minds Programme - Batch Processing Model
Research by: Zoe Akua Ohene-Ampofo
University of Ghana, MSc Business Analytics

This script implements the predictive model for batch processing of district data
and can be used independently of the Streamlit interface.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class GESLivelyMindsPredictor:
    """
    Predictive model for GES Lively Minds Programme success assessment
    
    Based on methodology from Chapter 3: Research Methodology
    - Uses logistic regression with L2 regularization
    - Implements feature engineering as specified
    - Provides district-level predictions for expansion planning
    """
    
    def __init__(self, random_state=42):
        self.model = LogisticRegression(
            random_state=random_state,
            solver='liblinear',
            penalty='l2',
            max_iter=1000
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        self.performance_metrics = {}
        
    def preprocess_data(self, df, target_column='success'):
        """
        Complete data preprocessing pipeline following Chapter 3 methodology
        
        Args:
            df: Input DataFrame with raw district data
            target_column: Name of target column (default: 'success')
            
        Returns:
            tuple: (X_processed, y, df_processed)
        """
        print("Starting data preprocessing...")
        
        # Clean data - remove empty or invalid records
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=['District', 'Recording_Completion', 'Team_Preparation', 
                                          'Workshop_Delivery', 'Team_Motivation'])
        df_clean = df_clean[df_clean['District'].str.strip() != '']
        
        print(f"Data cleaned: {len(df_clean)} valid records from {len(df)} total")
        
        # Feature Engineering Step 1: Normalize and Score
        # Recording completion (0-100 scale, target = 15)
        df_clean['recording_score'] = np.clip((df_clean['Recording_Completion'] / 15) * 100, 0, 100)
        
        # Performance scoring maps
        performance_map = {'Excelled': 100, 'Got it': 85, 'Good': 70, 'Needs improvement': 55}
        motivation_map = {
            'Excited': 100, 'Excited, Motivated': 95, 'Motivated': 80,
            'Confident': 75, 'Neutral': 60, 'Worried/Concerned': 40
        }
        
        df_clean['preparation_score'] = df_clean['Team_Preparation'].map(performance_map).fillna(60)
        df_clean['delivery_score'] = df_clean['Workshop_Delivery'].map(performance_map).fillna(60)
        df_clean['motivation_score'] = df_clean['Team_Motivation'].map(motivation_map).fillna(60)
        
        # Composite score calculation (weighted as per methodology)
        df_clean['composite_score'] = (
            0.4 * df_clean['recording_score'] + 
            0.2 * df_clean['preparation_score'] + 
            0.2 * df_clean['delivery_score'] + 
            0.2 * df_clean['motivation_score']
        )
        
        # Binary classification (success >= 80)
        if target_column not in df_clean.columns:
            df_clean['success'] = (df_clean['composite_score'] >= 80).astype(int)
        
        # Feature Engineering Step 2: One-hot encoding
        # Preparation categories
        prep_dummies = pd.get_dummies(df_clean['Team_Preparation'], prefix='prep')
        delivery_dummies = pd.get_dummies(df_clean['Workshop_Delivery'], prefix='delivery')
        motivation_dummies = pd.get_dummies(df_clean['Team_Motivation'], prefix='motivation')
        
        # Sentiment analysis on comments
        df_clean['sentiment_score'] = df_clean['Comments'].fillna('').apply(self._calculate_sentiment)
        df_clean['comment_length'] = df_clean['Comments'].fillna('').str.len()
        
        # Combine all features
        feature_columns = ['recording_score', 'preparation_score', 'delivery_score', 
                          'motivation_score', 'sentiment_score', 'comment_length']
        
        X_numerical = df_clean[feature_columns]
        X_categorical = pd.concat([prep_dummies, delivery_dummies, motivation_dummies], axis=1)
        X_processed = pd.concat([X_numerical, X_categorical], axis=1)
        
        # Store feature column names for future use
        self.feature_columns = X_processed.columns.tolist()
        
        y = df_clean[target_column] if target_column in df_clean.columns else df_clean['success']
        
        print(f"Feature engineering completed: {len(X_processed.columns)} features created")
        print(f"Success rate: {y.mean():.1%}")
        
        return X_processed, y, df_clean
    
    def _calculate_sentiment(self, text):
        """Simple lexicon-based sentiment analysis"""
        if pd.isna(text) or text == '':
            return 0
        
        positive_words = ['good', 'great', 'excellent', 'excited', 'motivated', 'ready', 
                         'happy', 'best', 'wonderful', 'amazing', 'satisfied', 'pleased',
                         'successful', 'effective', 'positive', 'strong']
        
        negative_words = ['bad', 'poor', 'terrible', 'worried', 'concerned', 'problem', 
                         'issue', 'difficult', 'hard', 'inadequate', 'insufficient', 
                         'lack', 'failed', 'weak', 'disappointed', 'frustrated']
        
        words = str(text).lower().split()
        if len(words) == 0:
            return 0
            
        positive_count = sum(1 for word in words if any(pos in word for pos in positive_words))
        negative_count = sum(1 for word in words if any(neg in word for neg in negative_words))
        
        return (positive_count - negative_count) / len(words)
    
    def train_model(self, X, y, test_size=0.3, cv_folds=5):
        """
        Train the logistic regression model with validation
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing (default: 0.3)
            cv_folds: Number of cross-validation folds (default: 5)
            
        Returns:
            dict: Training results and performance metrics
        """
        print("Training logistic regression model...")
        
        # Train-test split (stratified to preserve class distribution)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                   cv=cv_folds, scoring='accuracy')
        
        # Store performance metrics
        self.performance_metrics = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        # Print results
        print(f"\nModel Training Results:")
        print(f"Training Accuracy: {train_metrics['accuracy']:.3f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
        print(f"Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Check performance thresholds
        self._check_performance_thresholds(test_metrics)
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'classification_report': classification_report(y_test, y_test_pred)
        }
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive performance metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        }
    
    def _check_performance_thresholds(self, metrics):
        """Check if model meets performance thresholds from methodology"""
        thresholds = {
            'accuracy': 0.80,
            'precision': 0.75,
            'recall': 0.80,
            'f1': 0.77,
            'specificity': 0.75
        }
        
        print("\nPerformance Threshold Analysis:")
        all_passed = True
        
        for metric, threshold in thresholds.items():
            value = metrics[metric]
            status = "PASSED" if value >= threshold else "NEEDS IMPROVEMENT"
            print(f"{metric.capitalize()}: {value:.3f} (threshold: {threshold}) - {status}")
            if value < threshold:
                all_passed = False
        
        if all_passed:
            print("Model meets all deployment thresholds!")
        else:
            print("Model needs improvement before deployment")
    
    def predict_district_success(self, district_data):
        """
        Predict success probability for new districts
        
        Args:
            district_data: DataFrame with district information
            
        Returns:
            DataFrame: Predictions with probabilities and recommendations
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess input data
        X_processed, _, df_processed = self.preprocess_data(district_data, target_column=None)
        
        # Ensure feature alignment
        for col in self.feature_columns:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        X_processed = X_processed[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Create results DataFrame
        results = df_processed[['District', 'composite_score']].copy()
        results['predicted_success'] = predictions
        results['success_probability'] = probabilities[:, 1]
        results['needs_support_probability'] = probabilities[:, 0]
        
        # Add recommendations
        results['recommendation'] = results.apply(self._generate_recommendation, axis=1)
        results['priority_level'] = results['success_probability'].apply(self._assign_priority)
        
        return results
    
    def _generate_recommendation(self, row):
        """Generate district-specific recommendations"""
        if row['success_probability'] >= 0.8:
            return "Excellent performance - suitable for expansion model"
        elif row['success_probability'] >= 0.6:
            return "Good potential - provide moderate support"
        elif row['success_probability'] >= 0.4:
            return "Needs targeted intervention - focus on weak areas"
        else:
            return "High risk - requires intensive support before expansion"
    
    def _assign_priority(self, prob):
        """Assign priority level based on success probability"""
        if prob >= 0.8:
            return "Low"
        elif prob >= 0.6:
            return "Medium"
        elif prob >= 0.4:
            return "High"
        else:
            return "Critical"
    
    def save_model(self, filepath):
        """Save trained model and scaler"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'performance_metrics': self.performance_metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and scaler"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.performance_metrics = model_data.get('performance_metrics', {})
        self.is_fitted = True
        
        print(f"Model loaded from {filepath}")
    
    def generate_district_report(self, predictions_df, output_file=None):
        """Generate comprehensive district analysis report"""
        
        report = {
            'summary': {
                'total_districts': len(predictions_df),
                'predicted_successful': (predictions_df['predicted_success'] == 1).sum(),
                'success_rate': predictions_df['predicted_success'].mean(),
                'avg_success_probability': predictions_df['success_probability'].mean(),
                'high_priority_districts': (predictions_df['priority_level'] == 'High').sum(),
                'critical_districts': (predictions_df['priority_level'] == 'Critical').sum()
            },
            'top_performers': predictions_df.nlargest(10, 'success_probability')[
                ['District', 'composite_score', 'success_probability', 'recommendation']
            ],
            'needs_support': predictions_df.nsmallest(10, 'success_probability')[
                ['District', 'composite_score', 'success_probability', 'recommendation']
            ],
            'priority_distribution': predictions_df['priority_level'].value_counts(),
            'recommendation_distribution': predictions_df['recommendation'].value_counts()
        }
        
        if output_file:
            # Save detailed report to Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                predictions_df.to_excel(writer, sheet_name='District_Predictions', index=False)
                
                # Summary sheet
                summary_df = pd.DataFrame([report['summary']]).T
                summary_df.columns = ['Value']
                summary_df.to_excel(writer, sheet_name='Summary')
                
                # Top performers and needs support
                report['top_performers'].to_excel(writer, sheet_name='Top_Performers', index=False)
                report['needs_support'].to_excel(writer, sheet_name='Needs_Support', index=False)
            
            print(f"Detailed report saved to {output_file}")
        
        return report


def main():
    """Example usage of the GES Lively Minds Predictor"""
    
    print("GES Lively Minds Programme - Predictive Model")
    print("=" * 50)
    
    # Example: Load your data
    # df = pd.read_excel('LMT_Topup_Long essay_final.xlsx')
    
    # For demonstration, create sample data
    np.random.seed(42)
    sample_data = {
        'District': [f'District_{i:03d}' for i in range(1, 101)],
        'Recording_Completion': np.random.randint(8, 16, 100),
        'Team_Preparation': np.random.choice(['Excelled', 'Got it', 'Good', 'Needs improvement'], 100),
        'Workshop_Delivery': np.random.choice(['Excelled', 'Got it', 'Good', 'Needs improvement'], 100),
        'Team_Motivation': np.random.choice(['Excited', 'Motivated', 'Confident', 'Neutral', 'Worried/Concerned'], 100),
        'Comments': ['Good progress'] * 100
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Sample data created: {len(df)} districts")
    
    # Initialize and train model
    predictor = GESLivelyMindsPredictor()
    
    # Preprocess data
    X, y, df_processed = predictor.preprocess_data(df)
    
    # Train model
    results = predictor.train_model(X, y)
    
    # Make predictions on new data (using same data for demo)
    predictions = predictor.predict_district_success(df)
    
    print("\nSample Predictions:")
    print(predictions[['District', 'composite_score', 'success_probability', 'recommendation']].head())
    
    # Generate report
    report = predictor.generate_district_report(predictions)
    
    print(f"\nReport Summary:")
    print(f"Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"High Priority Districts: {report['summary']['high_priority_districts']}")
    print(f"Critical Districts: {report['summary']['critical_districts']}")
    
    # Save model for future use
    predictor.save_model('ges_lmt_model.pkl')


if __name__ == "__main__":
    main()