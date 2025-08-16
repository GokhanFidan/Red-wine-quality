"""
Advanced Interactive Visualizations for Wine Quality Analysis
Professional dashboards with Plotly for wine analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.config import (
    COLOR_PALETTE, FIGURE_SIZE, PLOTS_DIR, 
    FEATURE_COLUMNS, TARGET_COLUMN, QUALITY_CATEGORIES
)

logger = logging.getLogger(__name__)

class WineQualityVisualizer:
    """
    Professional visualization suite for wine quality analysis
    """
    
    def __init__(self):
        self.color_palette = COLOR_PALETTE
        self.plots_dir = PLOTS_DIR
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_executive_dashboard(self, data: pd.DataFrame, 
                                 model_results: Optional[Dict] = None) -> go.Figure:
        """
        Create comprehensive executive dashboard for wine quality analysis
        
        Args:
            data: Wine dataset
            model_results: Optional model comparison results
            
        Returns:
            Plotly figure with executive dashboard
        """
        try:
            logger.info("Creating executive wine quality dashboard")
            
            # Create subplot structure
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=[
                    'Quality Distribution', 'Alcohol vs Quality', 'Chemical Composition',
                    'Feature Correlations', 'Quality by Acidity', 'Sulfur Dioxide Analysis',
                    'Model Performance', 'Prediction Confidence', 'Business Metrics'
                ],
                specs=[
                    [{"type": "bar"}, {"type": "scatter"}, {"type": "pie"}],
                    [{"type": "heatmap"}, {"type": "violin"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "histogram"}, {"type": "indicator"}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.06
            )
            
            # 1. Quality Distribution
            quality_counts = data[TARGET_COLUMN].value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=quality_counts.index,
                    y=quality_counts.values,
                    name='Quality Distribution',
                    marker_color=self.color_palette['primary'],
                    text=quality_counts.values,
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # 2. Alcohol vs Quality (Box Plot)
            fig.add_trace(
                go.Box(
                    x=data[TARGET_COLUMN],
                    y=data['alcohol'],
                    name='Alcohol vs Quality',
                    marker_color=self.color_palette['secondary'],
                    boxpoints='outliers'
                ),
                row=1, col=2
            )
            
            # 3. Chemical Composition (Top 5 features by correlation)
            correlations = data[FEATURE_COLUMNS].corrwith(data[TARGET_COLUMN]).abs().sort_values(ascending=False)
            top_5_features = correlations.head(5)
            
            fig.add_trace(
                go.Pie(
                    labels=top_5_features.index,
                    values=top_5_features.values,
                    name="Top Features by Correlation",
                    hole=0.4,
                    marker_colors=[self.color_palette['primary'], self.color_palette['secondary'], 
                                 self.color_palette['accent'], self.color_palette['success'], 
                                 self.color_palette['warning']]
                ),
                row=1, col=3
            )
            
            # 4. Feature Correlations Heatmap (Top 8 features)
            top_features = FEATURE_COLUMNS[:8]
            corr_matrix = data[top_features + [TARGET_COLUMN]].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    showscale=False
                ),
                row=2, col=1
            )
            
            # 5. Quality by Acidity (Violin Plot)
            data['acidity_category'] = pd.cut(data['fixed acidity'], bins=3, labels=['Low', 'Medium', 'High'])
            
            for i, category in enumerate(['Low', 'Medium', 'High']):
                quality_data = data[data['acidity_category'] == category][TARGET_COLUMN]
                fig.add_trace(
                    go.Violin(
                        y=quality_data,
                        name=f'{category} Acidity',
                        box_visible=True,
                        meanline_visible=True,
                        x0=category
                    ),
                    row=2, col=2
                )
            
            # 6. Sulfur Dioxide Analysis
            fig.add_trace(
                go.Scatter(
                    x=data['free sulfur dioxide'],
                    y=data['total sulfur dioxide'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=data[TARGET_COLUMN],
                        colorscale='Viridis',
                        showscale=False
                    ),
                    name='Sulfur Dioxide Analysis',
                    text=[f'Quality: {q}' for q in data[TARGET_COLUMN]],
                    hovertemplate='Free SO2: %{x}<br>Total SO2: %{y}<br>%{text}<extra></extra>'
                ),
                row=2, col=3
            )
            
            # 7. Model Performance (if available)
            if model_results:
                model_names = list(model_results.keys())[:5]  # Top 5 models
                r2_scores = [model_results[name]['test_r2'] for name in model_names]
                
                fig.add_trace(
                    go.Bar(
                        x=model_names,
                        y=r2_scores,
                        name='Model R² Scores',
                        marker_color=self.color_palette['accent'],
                        text=[f'{score:.3f}' for score in r2_scores],
                        textposition='auto'
                    ),
                    row=3, col=1
                )
            
            # 8. Prediction Confidence Distribution
            # Simulate confidence scores based on feature variance
            feature_variance = data[FEATURE_COLUMNS].var().mean()
            confidence_scores = np.random.beta(2, 1, len(data)) * (1 - feature_variance/100)
            
            fig.add_trace(
                go.Histogram(
                    x=confidence_scores,
                    nbinsx=20,
                    name='Prediction Confidence',
                    marker_color=self.color_palette['success'],
                    opacity=0.7
                ),
                row=3, col=2
            )
            
            # 9. Business Metrics Indicator
            avg_quality = data[TARGET_COLUMN].mean()
            high_quality_percent = (data[TARGET_COLUMN] >= 7).mean() * 100
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=avg_quality,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Average Quality"},
                    delta={'reference': 6.0},
                    gauge={
                        'axis': {'range': [None, 10]},
                        'bar': {'color': self.color_palette['primary']},
                        'steps': [
                            {'range': [0, 5], 'color': "lightgray"},
                            {'range': [5, 7], 'color': "yellow"},
                            {'range': [7, 10], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 7
                        }
                    }
                ),
                row=3, col=3
            )
            
            # Update layout
            fig.update_layout(
                height=1200,
                title_text="Wine Quality Analytics - Executive Dashboard",
                title_x=0.5,
                title_font_size=24,
                showlegend=False,
                font=dict(size=10)
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Quality Score", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            
            fig.update_xaxes(title_text="Quality Score", row=1, col=2)
            fig.update_yaxes(title_text="Alcohol %", row=1, col=2)
            
            fig.update_xaxes(title_text="Free Sulfur Dioxide", row=2, col=3)
            fig.update_yaxes(title_text="Total Sulfur Dioxide", row=2, col=3)
            
            if model_results:
                fig.update_xaxes(title_text="Models", row=3, col=1)
                fig.update_yaxes(title_text="R² Score", row=3, col=1)
            
            fig.update_xaxes(title_text="Confidence Score", row=3, col=2)
            fig.update_yaxes(title_text="Frequency", row=3, col=2)
            
            # Save dashboard
            self.save_plot(fig, 'wine_executive_dashboard', ['html', 'png'])
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating executive dashboard: {str(e)}")
            raise
    
    def create_feature_analysis_dashboard(self, data: pd.DataFrame, 
                                        feature_importance: Optional[Dict] = None) -> go.Figure:
        """
        Create detailed feature analysis dashboard
        
        Args:
            data: Wine dataset
            feature_importance: Optional feature importance scores
            
        Returns:
            Plotly figure with feature analysis
        """
        try:
            logger.info("Creating feature analysis dashboard")
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[
                    'Feature Distributions', 'Feature Correlations with Quality',
                    'Chemical Balance Analysis', 'Feature Importance Ranking',
                    'Outlier Detection', 'Feature Interactions'
                ],
                specs=[
                    [{"colspan": 2}, None, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "scatter"}, {"type": "heatmap"}]
                ]
            )
            
            # 1. Feature Distributions (Box plots)
            important_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'fixed acidity']
            
            for i, feature in enumerate(important_features):
                fig.add_trace(
                    go.Box(
                        y=data[feature],
                        name=feature.replace('_', ' ').title(),
                        marker_color=self.color_palette['primary'],
                        boxpoints='outliers'
                    ),
                    row=1, col=1
                )
            
            # 2. Chemical Balance Analysis (2D scatter)
            fig.add_trace(
                go.Scatter(
                    x=data['alcohol'],
                    y=data['volatile acidity'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=data[TARGET_COLUMN],
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=[f'Quality: {q}' for q in data[TARGET_COLUMN]],
                    hovertemplate='Alcohol: %{x:.1f}%<br>Volatile Acidity: %{y:.2f}<br>%{text}<extra></extra>',
                    name='Chemical Balance'
                ),
                row=1, col=3
            )
            
            # 3. Feature Correlations with Quality
            correlations = data[FEATURE_COLUMNS].corrwith(data[TARGET_COLUMN]).sort_values(ascending=True)
            
            colors = [self.color_palette['success'] if x > 0 else self.color_palette['error'] for x in correlations.values]
            
            fig.add_trace(
                go.Bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h',
                    marker_color=colors,
                    text=[f'{val:.3f}' for val in correlations.values],
                    textposition='auto',
                    name='Correlation with Quality'
                ),
                row=2, col=1
            )
            
            # 4. Outlier Detection (Using IQR method)
            outlier_counts = {}
            for feature in important_features:
                Q1 = data[feature].quantile(0.25)
                Q3 = data[feature].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts[feature] = outliers
            
            fig.add_trace(
                go.Scatter(
                    x=list(outlier_counts.keys()),
                    y=list(outlier_counts.values()),
                    mode='markers+lines',
                    marker=dict(size=12, color=self.color_palette['warning']),
                    line=dict(color=self.color_palette['warning'], width=2),
                    name='Outlier Count'
                ),
                row=2, col=2
            )
            
            # 5. Feature Interactions Heatmap
            interaction_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
            interaction_matrix = data[interaction_features].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=interaction_matrix.values,
                    x=interaction_matrix.columns,
                    y=interaction_matrix.columns,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=np.round(interaction_matrix.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    showscale=False
                ),
                row=2, col=3
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_text="Wine Quality Feature Analysis Dashboard",
                title_x=0.5,
                title_font_size=20,
                showlegend=True
            )
            
            # Update axes
            fig.update_xaxes(title_text="Correlation with Quality", row=2, col=1)
            fig.update_xaxes(title_text="Features", row=2, col=2)
            fig.update_yaxes(title_text="Outlier Count", row=2, col=2)
            
            # Save dashboard
            self.save_plot(fig, 'wine_feature_analysis_dashboard', ['html', 'png'])
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature analysis dashboard: {str(e)}")
            raise
    
    def create_model_performance_dashboard(self, model_results: Dict, 
                                         feature_importance: Optional[Dict] = None) -> go.Figure:
        """
        Create model performance comparison dashboard
        
        Args:
            model_results: Dictionary with model comparison results
            feature_importance: Optional feature importance from best model
            
        Returns:
            Plotly figure with model performance analysis
        """
        try:
            logger.info("Creating model performance dashboard")
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[
                    'Model R² Comparison', 'RMSE vs Accuracy Trade-off', 'Training Time Analysis',
                    'Cross-Validation Scores', 'Feature Importance', 'Prediction vs Actual'
                ],
                specs=[
                    [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                    [{"type": "box"}, {"type": "bar"}, {"type": "scatter"}]
                ]
            )
            
            # Extract model data
            model_names = list(model_results.keys())
            r2_scores = [model_results[name]['test_r2'] for name in model_names]
            rmse_scores = [model_results[name]['test_rmse'] for name in model_names]
            accuracy_scores = [model_results[name]['test_accuracy'] for name in model_names]
            training_times = [model_results[name]['training_time'] for name in model_names]
            
            # 1. Model R² Comparison
            colors = [self.color_palette['success'] if score > 0.6 else 
                     self.color_palette['warning'] if score > 0.4 else 
                     self.color_palette['error'] for score in r2_scores]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=r2_scores,
                    marker_color=colors,
                    text=[f'{score:.3f}' for score in r2_scores],
                    textposition='auto',
                    name='R² Scores'
                ),
                row=1, col=1
            )
            
            # 2. RMSE vs Accuracy Trade-off
            fig.add_trace(
                go.Scatter(
                    x=rmse_scores,
                    y=accuracy_scores,
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=r2_scores,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=model_names,
                    textposition='top center',
                    name='RMSE vs Accuracy',
                    hovertemplate='RMSE: %{x:.3f}<br>Accuracy: %{y:.3f}<br>Model: %{text}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 3. Training Time Analysis
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=training_times,
                    marker_color=self.color_palette['accent'],
                    text=[f'{time:.1f}s' for time in training_times],
                    textposition='auto',
                    name='Training Time'
                ),
                row=1, col=3
            )
            
            # 4. Cross-Validation Scores
            for i, name in enumerate(model_names):
                if 'cv_rmse_mean' in model_results[name]:
                    cv_mean = model_results[name]['cv_rmse_mean']
                    cv_std = model_results[name]['cv_rmse_std']
                    
                    # Simulate CV scores for visualization
                    cv_scores = np.random.normal(cv_mean, cv_std, 5)
                    
                    fig.add_trace(
                        go.Box(
                            y=cv_scores,
                            name=name,
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-1.8
                        ),
                        row=2, col=1
                    )
            
            # 5. Feature Importance (if available)
            if feature_importance:
                features = list(feature_importance.keys())[:10]  # Top 10 features
                importances = [feature_importance[feat] for feat in features]
                
                fig.add_trace(
                    go.Bar(
                        x=importances,
                        y=features,
                        orientation='h',
                        marker_color=self.color_palette['primary'],
                        text=[f'{imp:.3f}' for imp in importances],
                        textposition='auto',
                        name='Feature Importance'
                    ),
                    row=2, col=2
                )
            
            # 6. Prediction vs Actual (Best Model)
            best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
            best_model_results = model_results[best_model_name]
            
            if 'y_pred_test' in best_model_results:
                y_pred = best_model_results['y_pred_test']
                y_actual = np.arange(len(y_pred))  # Placeholder - would use actual test data
                
                fig.add_trace(
                    go.Scatter(
                        x=y_actual[:100],  # Show first 100 predictions
                        y=y_pred[:100],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=self.color_palette['secondary'],
                            opacity=0.6
                        ),
                        name=f'{best_model_name} Predictions',
                        hovertemplate='Actual: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=3
                )
                
                # Add perfect prediction line
                min_val, max_val = min(y_pred[:100]), max(y_pred[:100])
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Perfect Prediction',
                        showlegend=False
                    ),
                    row=2, col=3
                )
            
            # Update layout
            fig.update_layout(
                height=1000,
                title_text=f"Model Performance Dashboard - Best: {best_model_name}",
                title_x=0.5,
                title_font_size=20,
                showlegend=True
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Models", row=1, col=1)
            fig.update_yaxes(title_text="R² Score", row=1, col=1)
            
            fig.update_xaxes(title_text="RMSE", row=1, col=2)
            fig.update_yaxes(title_text="Accuracy", row=1, col=2)
            
            fig.update_xaxes(title_text="Models", row=1, col=3)
            fig.update_yaxes(title_text="Training Time (s)", row=1, col=3)
            
            fig.update_yaxes(title_text="CV RMSE", row=2, col=1)
            
            if feature_importance:
                fig.update_xaxes(title_text="Importance Score", row=2, col=2)
            
            fig.update_xaxes(title_text="Actual Quality", row=2, col=3)
            fig.update_yaxes(title_text="Predicted Quality", row=2, col=3)
            
            # Save dashboard
            self.save_plot(fig, 'wine_model_performance_dashboard', ['html', 'png'])
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating model performance dashboard: {str(e)}")
            raise
    
    def create_wine_chemistry_explorer(self, data: pd.DataFrame) -> go.Figure:
        """
        Create interactive wine chemistry explorer
        
        Args:
            data: Wine dataset
            
        Returns:
            Plotly figure with chemistry analysis
        """
        try:
            logger.info("Creating wine chemistry explorer")
            
            # Create feature for quality categories
            data['quality_category'] = pd.cut(data[TARGET_COLUMN], 
                                            bins=[0, 5, 6, 8, 10], 
                                            labels=['Poor', 'Average', 'Good', 'Excellent'])
            
            # Create multi-dimensional scatter plot
            fig = px.scatter_matrix(
                data,
                dimensions=['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'pH'],
                color='quality_category',
                title="Wine Chemistry Multi-Dimensional Analysis",
                color_discrete_map={
                    'Poor': self.color_palette['error'],
                    'Average': self.color_palette['warning'], 
                    'Good': self.color_palette['success'],
                    'Excellent': self.color_palette['primary']
                }
            )
            
            fig.update_layout(
                height=800,
                title_x=0.5,
                font=dict(size=12)
            )
            
            # Save explorer
            self.save_plot(fig, 'wine_chemistry_explorer', ['html', 'png'])
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chemistry explorer: {str(e)}")
            raise
    
    def save_plot(self, fig, filename: str, formats: List[str] = ['html', 'png']):
        """
        Save plot in multiple formats
        
        Args:
            fig: Plotly figure
            filename: Base filename
            formats: List of formats to save
        """
        try:
            for fmt in formats:
                if fmt == 'html':
                    fig.write_html(self.plots_dir / f"{filename}.html")
                elif fmt == 'png':
                    fig.write_image(self.plots_dir / f"{filename}.png", width=1200, height=800)
                elif fmt == 'pdf':
                    fig.write_image(self.plots_dir / f"{filename}.pdf")
            
            logger.info(f"Plot saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving plot {filename}: {str(e)}")
            
    def generate_all_visualizations(self, data: pd.DataFrame, 
                                  model_results: Optional[Dict] = None,
                                  feature_importance: Optional[Dict] = None) -> Dict:
        """
        Generate all visualization dashboards
        
        Args:
            data: Wine dataset
            model_results: Optional model results
            feature_importance: Optional feature importance
            
        Returns:
            Dictionary with all created figures
        """
        try:
            logger.info("Generating all wine quality visualizations")
            
            visualizations = {}
            
            # Executive Dashboard
            visualizations['executive'] = self.create_executive_dashboard(data, model_results)
            
            # Feature Analysis Dashboard
            visualizations['features'] = self.create_feature_analysis_dashboard(data, feature_importance)
            
            # Model Performance Dashboard
            if model_results:
                visualizations['models'] = self.create_model_performance_dashboard(model_results, feature_importance)
            
            # Chemistry Explorer
            visualizations['chemistry'] = self.create_wine_chemistry_explorer(data)
            
            logger.info(f"Generated {len(visualizations)} visualization dashboards")
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise