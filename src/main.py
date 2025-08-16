"""
Wine Quality Prediction - Main Pipeline
Professional ML pipeline for wine quality analysis and prediction
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from data_processing.data_loader import WineDataLoader
from features.feature_engineering import WineFeatureEngineer
from models.wine_models import WineQualityModelSuite
from visualization.wine_visualizer import WineQualityVisualizer

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config.config import LOGGING_CONFIG, MODELS_DIR, PLOTS_DIR

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_section_header(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

def print_results_summary(results: dict):
    """Print formatted results summary"""
    print(f"\n📊 Analysis Results:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"   • {key}: {value:.4f}")
            else:
                print(f"   • {key}: {value:,}")
        else:
            print(f"   • {key}: {value}")

def main():
    """Main execution pipeline"""
    try:
        print_section_header("WINE QUALITY PREDICTION - ENTERPRISE ML PIPELINE")
        print("🍷 Professional wine quality analysis and prediction system")
        print("🎯 Designed for portfolio demonstration and business applications")
        
        # 1. Data Loading and Preprocessing
        print_section_header("DATA LOADING & PREPROCESSING")
        print("📁 Loading and validating wine quality dataset...")
        
        data_loader = WineDataLoader()
        features, target = data_loader.load_and_prepare_data()
        
        print(f"✅ Data loaded successfully:")
        print(f"   • Features shape: {features.shape}")
        print(f"   • Target shape: {target.shape}")
        print(f"   • Quality distribution: {target.value_counts().sort_index().to_dict()}")
        
        # Validation report summary
        if hasattr(data_loader, 'validation_report'):
            validation = data_loader.validation_report
            print(f"   • Validation status: {'✅ Passed' if validation['is_valid'] else '❌ Failed'}")
            if validation['warnings']:
                print(f"   • Warnings: {len(validation['warnings'])}")
        
        # 2. Feature Engineering
        print_section_header("ADVANCED FEATURE ENGINEERING")
        print("🔧 Creating domain-specific wine chemistry features...")
        
        feature_engineer = WineFeatureEngineer()
        
        # Create engineered features
        features_engineered = feature_engineer.create_domain_features(features)
        print(f"✅ Feature engineering completed:")
        print(f"   • Original features: {len(features.columns)}")
        print(f"   • Engineered features: {len(feature_engineer.engineered_features)}")
        print(f"   • Total features: {len(features_engineered.columns)}")
        
        # Feature importance analysis
        importance_analysis = feature_engineer.get_feature_importance_analysis(features_engineered, target)
        
        # Get top features by correlation
        top_features_by_corr = sorted(importance_analysis['correlations'].items(), 
                                    key=lambda x: abs(x[1]), reverse=True)[:10]
        
        print(f"\n🔝 Top 10 Features by Correlation with Quality:")
        for i, (feature, corr) in enumerate(top_features_by_corr, 1):
            print(f"   {i:2d}. {feature}: {corr:.4f}")
        
        # 3. Model Training and Evaluation
        print_section_header("MACHINE LEARNING MODEL TRAINING")
        print("🤖 Training and comparing multiple ML algorithms...")
        
        # Split data for training
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features_engineered, target, test_size=0.2, random_state=42, stratify=target
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = feature_engineer.scale_features(X_train, X_test, method='standard')
        
        # Feature selection
        X_train_selected, X_test_selected = feature_engineer.select_features(
            X_train_scaled, y_train, X_test_scaled, n_features=20
        )
        
        print(f"✅ Data preparation for modeling:")
        print(f"   • Training samples: {X_train_selected.shape[0]:,}")
        print(f"   • Test samples: {X_test_selected.shape[0]:,}")
        print(f"   • Selected features: {X_train_selected.shape[1]}")
        
        # Initialize and train models
        model_suite = WineQualityModelSuite()
        model_suite.initialize_models()
        
        print(f"🏭 Training {len(model_suite.models)} machine learning models...")
        
        # Train all models
        model_results = model_suite.train_all_models(
            X_train_selected, y_train, X_test_selected, y_test,
            hyperparameter_tuning=True
        )
        
        print(f"✅ Model training completed:")
        print(f"   • Models trained: {len(model_results)}")
        print(f"   • Best model: {model_suite.best_model_name}")
        print(f"   • Best R² score: {model_results[model_suite.best_model_name]['test_r2']:.4f}")
        
        # Create ensemble models
        print("\n🎭 Creating ensemble models...")
        ensemble_results = model_suite.create_ensemble_models(
            X_train_selected, y_train, X_test_selected, y_test
        )
        
        if ensemble_results:
            print(f"✅ Ensemble models created: {list(ensemble_results.keys())}")
        
        # Model comparison
        comparison_df = model_suite.get_model_comparison()
        print(f"\n📈 Model Performance Ranking:")
        print(comparison_df.head(5).to_string(index=False, float_format='%.4f'))
        
        # Save best model
        model_suite.save_best_model()
        print(f"💾 Best model saved: {model_suite.best_model_name}")
        
        # 4. Advanced Visualizations
        print_section_header("INTERACTIVE VISUALIZATIONS & DASHBOARDS")
        print("📊 Creating professional interactive dashboards...")
        
        # Combine original data with target for visualization
        viz_data = features.copy()
        viz_data['quality'] = target
        
        # Get feature importance from best model
        best_model_results = model_results[model_suite.best_model_name]
        feature_importance = best_model_results.get('feature_importance')
        
        # Create visualizations
        visualizer = WineQualityVisualizer()
        all_visualizations = visualizer.generate_all_visualizations(
            viz_data, model_results, feature_importance
        )
        
        print(f"✅ Visualizations created:")
        for viz_name, fig in all_visualizations.items():
            print(f"   • {viz_name.title()} Dashboard")
        
        print(f"   • Saved to: {PLOTS_DIR}")
        
        # 5. Business Insights & Recommendations
        print_section_header("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("💼 Generating data-driven business recommendations...")
        
        # Calculate business metrics
        total_wines = len(viz_data)
        high_quality_wines = (viz_data['quality'] >= 7).sum()
        high_quality_percentage = (high_quality_wines / total_wines) * 100
        avg_alcohol = viz_data['alcohol'].mean()
        
        # Feature insights
        top_quality_features = viz_data[viz_data['quality'] >= 7][list(features.columns)].mean()
        low_quality_features = viz_data[viz_data['quality'] <= 5][list(features.columns)].mean()
        
        print(f"\n📊 Business Metrics:")
        print(f"   • Total wine samples analyzed: {total_wines:,}")
        print(f"   • High quality wines (≥7): {high_quality_wines:,} ({high_quality_percentage:.1f}%)")
        print(f"   • Average alcohol content: {avg_alcohol:.1f}%")
        print(f"   • Model prediction accuracy: {best_model_results['test_accuracy']*100:.1f}%")
        
        print(f"\n🔍 Key Quality Drivers:")
        quality_drivers = (top_quality_features - low_quality_features).abs().sort_values(ascending=False).head(5)
        for i, (feature, diff) in enumerate(quality_drivers.items(), 1):
            direction = "higher" if (top_quality_features[feature] > low_quality_features[feature]) else "lower"
            print(f"   {i}. {feature.title()}: {direction} values correlate with quality")
        
        print(f"\n💡 Business Recommendations:")
        recommendations = [
            "🎯 Focus quality control on alcohol content and volatile acidity levels",
            "🔬 Implement automated quality prediction to reduce manual testing costs",
            "📈 Use model insights to optimize wine production processes",
            "🏆 Target premium market segments with high-quality wine predictions",
            "⚡ Deploy real-time quality monitoring in production pipelines"
        ]
        
        for rec in recommendations:
            print(f"   • {rec}")
        
        # 6. Production Readiness
        print_section_header("PRODUCTION DEPLOYMENT READINESS")
        
        # Check model performance against thresholds
        from config.config import PERFORMANCE_THRESHOLDS
        
        best_performance = model_results[model_suite.best_model_name]
        
        performance_check = {
            'Accuracy': (best_performance['test_accuracy'], PERFORMANCE_THRESHOLDS['min_accuracy']),
            'R² Score': (best_performance['test_r2'], PERFORMANCE_THRESHOLDS.get('min_r2', 0.7)),
            'Training Time': (best_performance['training_time'], PERFORMANCE_THRESHOLDS['max_training_time'])
        }
        
        print(f"🚀 Production Readiness Assessment:")
        all_passed = True
        
        for metric, (actual, threshold) in performance_check.items():
            if metric == 'Training Time':
                passed = actual <= threshold
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"   • {metric}: {actual:.1f}s (≤ {threshold}s) {status}")
            else:
                passed = actual >= threshold
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"   • {metric}: {actual:.3f} (≥ {threshold:.3f}) {status}")
            
            all_passed = all_passed and passed
        
        deployment_status = "🚀 READY FOR DEPLOYMENT" if all_passed else "⚠️  NEEDS OPTIMIZATION"
        print(f"\n{deployment_status}")
        
        # 7. Summary
        print_section_header("PIPELINE EXECUTION SUMMARY")
        
        pipeline_summary = {
            'Data Samples': total_wines,
            'Features Engineered': len(feature_engineer.engineered_features),
            'Models Trained': len(model_results),
            'Best Model': model_suite.best_model_name,
            'Best R² Score': best_model_results['test_r2'],
            'Prediction Accuracy': best_performance['test_accuracy'],
            'Visualizations Created': len(all_visualizations),
            'High Quality Wine %': high_quality_percentage
        }
        
        print_results_summary(pipeline_summary)
        
        print(f"\n📁 Generated Artifacts:")
        print(f"   • Models: {MODELS_DIR}")
        print(f"   • Visualizations: {PLOTS_DIR}")
        print(f"   • Processed Data: {data_loader.processed_df.shape if hasattr(data_loader, 'processed_df') else 'Available'}")
        
        print(f"\n🎉 Wine Quality Prediction Pipeline Completed Successfully!")
        print(f"📊 Ready for portfolio presentation and business deployment")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        print(f"\n❌ Pipeline Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)