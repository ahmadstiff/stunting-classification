"""
Main Pipeline Script for Binary Stunting Classification
Skripsi: Analisis Komparasi Kinerja Algoritma Ensemble Learning
         (Random Forest, XGBoost, LightGBM, dan CatBoost)
         untuk Klasifikasi Biner Status Stunting

Binary Classification:
    - 0 = Tidak Stunting (normal, tinggi)
    - 1 = Stunting (stunted, severely stunted)
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_RAW_PATH, MODELS_PATH, FIGURES_PATH, REPORTS_PATH,
    DATASET_NAME, RAW_DATA_FILE,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    CV_FOLDS, MODEL_NAMES, BINARY_CLASS_NAMES
)
from src.preprocessing import DataPreprocessor
from src.models import ModelTrainer, compare_models
from src.visualization import Visualizer


def download_dataset():
    """Download dataset from Kaggle using kagglehub."""
    print("\n" + "="*60)
    print("📥 DOWNLOADING DATASET")
    print("="*60)
    
    try:
        import kagglehub
        import shutil
        
        print(f"Downloading dataset: {DATASET_NAME}")
        path = kagglehub.dataset_download(DATASET_NAME)
        print(f"Dataset downloaded to: {path}")
        
        for file in os.listdir(path):
            if file.endswith('.csv'):
                src = os.path.join(path, file)
                dst = RAW_DATA_FILE
                shutil.copy(src, dst)
                print(f"Dataset copied to: {dst}")
                return dst
                
    except ImportError:
        print("kagglehub not installed. Please install with: pip install kagglehub")
        return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def run_eda(df, visualizer):
    """Run Exploratory Data Analysis and generate visualizations."""
    print("\nGenerating EDA visualizations...")
    
    # 1. Target distribution (original 4 classes)
    visualizer.plot_target_distribution(df)
    
    # 2. Numerical feature distributions
    num_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    if num_cols:
        visualizer.plot_numerical_distributions(df, num_cols)
        for col in num_cols:
            visualizer.plot_feature_by_target(df, col)
    
    # 3. Categorical feature distributions
    cat_cols = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    if cat_cols:
        visualizer.plot_categorical_distributions(df, cat_cols)
    
    # 4. Correlation heatmap
    if len(num_cols) > 1:
        visualizer.plot_correlation_heatmap(df, num_cols)


def train_single_model(model_name, trainer, X_train, y_train, X_test, y_test, 
                       feature_names, class_names, visualizer, tune_hyperparameters=True):
    """Train a single model and generate its visualizations."""
    
    print(f"\n{'─'*60}")
    print(f"🤖 Training {model_name}")
    print(f"{'─'*60}")
    
    # Create model
    model = trainer.create_model(model_name)
    
    # Hyperparameter tuning or direct training
    if tune_hyperparameters:
        param_grid = trainer.get_param_grid(model_name)
        model, best_params = trainer.hyperparameter_tuning(
            model, param_grid, X_train, y_train, model_name
        )
    else:
        model = trainer.train_model(model, X_train, y_train, model_name)
    
    # Cross-validation
    trainer.cross_validate(model, X_train, y_train, model_name, cv=CV_FOLDS)
    
    # Final training
    model = trainer.train_model(model, X_train, y_train, model_name)
    
    # Evaluation
    results = trainer.evaluate_model(model, X_test, y_test, model_name, class_names)
    
    # Feature importance
    importance = trainer.get_feature_importance(model, feature_names, model_name)
    
    # 📊 Visualizations for this model
    print(f"\n📊 Generating visualizations for {model_name}...")
    visualizer.plot_confusion_matrix(y_test, results['y_pred'], class_names, model_name)
    visualizer.plot_feature_importance(importance, model_name)
    
    return model, results, importance


def generate_comparison_visualizations(results, y_test, class_names, importances, trainer, visualizer):
    """Generate comparison visualizations between all models."""
    
    print("\n" + "═"*70)
    print("  📊 GENERATING COMPARISON VISUALIZATIONS")
    print("═"*70)
    
    # 1. Confusion matrix comparison
    print("\n📊 Confusion Matrix Comparison...")
    visualizer.plot_confusion_matrix_comparison(results, y_test, class_names)
    
    # 2. ROC curves (binary classification)
    print("\n📈 ROC Curves Comparison...")
    visualizer.plot_roc_curves_binary(results, y_test)
    
    # 3. Precision-Recall curves
    print("\n📈 Precision-Recall Curves...")
    visualizer.plot_precision_recall_curves(results, y_test)
    
    # 4. Metrics comparison
    print("\n📊 Metrics Comparison...")
    visualizer.plot_metrics_comparison(results)
    
    # 5. Cross-validation results
    if trainer.cv_results:
        print("\n📉 Cross-Validation Results...")
        visualizer.plot_cv_results(trainer.cv_results)
    
    # 6. Model comparison summary
    print("\n🏆 Model Comparison Summary Dashboard...")
    visualizer.plot_model_comparison_summary(results)
    
    # 7. Feature importance comparison (all 4 models)
    print("\n🔍 Feature Importance Comparison...")
    visualizer.plot_feature_importance_comparison_multi(importances)


def print_terminal_summary(results, class_names):
    """Print ASCII summary in terminal."""
    
    print("\n" + "="*80)
    print("  ╔══════════════════════════════════════════════════════════════════════════╗")
    print("  ║     HASIL KOMPARASI ALGORITMA ENSEMBLE LEARNING - BINARY STUNTING       ║")
    print("  ╚══════════════════════════════════════════════════════════════════════════╝")
    print("="*80)
    
    # Metrics comparison bar chart
    print("\n  📊 PERBANDINGAN METRIK:")
    print("  " + "─"*70)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    max_bar = 40
    
    for metric in metrics:
        print(f"\n  {metric.upper()}:")
        for model_name, result in results.items():
            val = result.get(metric, 0) or 0
            bar_len = int(val * max_bar)
            bar = "█" * bar_len + "░" * (max_bar - bar_len)
            
            if val >= 0.95:
                indicator = "🟢"
            elif val >= 0.90:
                indicator = "🟡"
            else:
                indicator = "🔴"
            
            print(f"    {indicator} {model_name:15} │{bar}│ {val:.4f} ({val*100:.2f}%)")
    
    # Ranking
    print("\n" + "  " + "─"*70)
    print("  🏆 RANKING (berdasarkan F1-Score):")
    print("  " + "─"*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('f1', 0), reverse=True)
    medals = ["🥇", "🥈", "🥉", "4️⃣"]
    
    for i, (model_name, result) in enumerate(sorted_results):
        f1 = result.get('f1', 0) * 100
        acc = result.get('accuracy', 0) * 100
        print(f"  {medals[i]} {model_name:15} - F1: {f1:.2f}% | Accuracy: {acc:.2f}%")
    
    # Winner
    winner = sorted_results[0][0]
    winner_f1 = sorted_results[0][1].get('f1', 0) * 100
    
    print(f"\n  ╔{'═'*70}╗")
    print(f"  ║  🏆 BEST MODEL: {winner:<52} ║")
    print(f"  ║  📊 F1-SCORE: {winner_f1:.2f}%{' '*(55-len(f'{winner_f1:.2f}'))}║")
    print(f"  ╚{'═'*70}╝")


def save_results(results, trainer):
    """Save results to CSV."""
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)
    
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1'],
            'ROC-AUC': result['roc_auc'],
            'Training Time (s)': result.get('training_time', 0)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    csv_path = REPORTS_PATH / 'model_comparison.csv'
    df_comparison.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    print(df_comparison.to_string(index=False))
    
    return df_comparison


def main(tune_hyperparameters=True, display_inline=True):
    """Main function to run the complete pipeline."""
    
    print("\n" + "="*80)
    print("  ╔══════════════════════════════════════════════════════════════════════════╗")
    print("  ║  KLASIFIKASI BINER STATUS STUNTING DENGAN ENSEMBLE LEARNING             ║")
    print("  ║  Random Forest | XGBoost | LightGBM | CatBoost                          ║")
    print("  ╚══════════════════════════════════════════════════════════════════════════╝")
    print("="*80)
    
    if display_inline:
        print("  🖼️  Inline image display: ENABLED")
    else:
        print("  🖼️  Inline image display: DISABLED")
    
    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    visualizer = Visualizer(display_inline=display_inline)
    
    # ================================================================
    # STEP 1: LOAD DATASET
    # ================================================================
    print("\n" + "═"*70)
    print("  STEP 1: LOAD DATASET")
    print("═"*70)
    
    if not RAW_DATA_FILE.exists():
        download_dataset()
    
    if not RAW_DATA_FILE.exists():
        print("\nERROR: Dataset not found!")
        return None, None
    
    df = preprocessor.load_data(RAW_DATA_FILE)
    
    # 📊 Data Overview
    print("\n📊 Visualisasi: Data Overview...")
    visualizer.plot_data_overview(df)
    
    # ================================================================
    # STEP 2: DATA EXPLORATION & CLEANING
    # ================================================================
    print("\n" + "═"*70)
    print("  STEP 2: DATA EXPLORATION & CLEANING")
    print("═"*70)
    
    df = preprocessor.explore_data(df)
    
    # 📊 Outlier Detection
    num_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    if num_cols:
        print("\n📊 Visualisasi: Outlier Detection...")
        visualizer.plot_outliers(df, num_cols)
    
    # Clean data
    initial_rows = len(df)
    df = preprocessor.clean_data(df)
    duplicates_removed = initial_rows - len(df)
    
    if duplicates_removed > 0:
        print("\n📊 Visualisasi: Duplicate Removal...")
        visualizer.plot_duplicates_analysis(initial_rows, len(df), duplicates_removed)
    
    # ================================================================
    # STEP 3: FEATURE ENCODING (BINARY)
    # ================================================================
    print("\n" + "═"*70)
    print("  STEP 3: FEATURE ENCODING (BINARY CLASSIFICATION)")
    print("═"*70)
    
    df_encoded = preprocessor.encode_features(df)
    
    # ================================================================
    # STEP 4: PREPARE FEATURES & SPLIT DATA
    # ================================================================
    print("\n" + "═"*70)
    print("  STEP 4: PREPARE FEATURES & SPLIT DATA")
    print("═"*70)
    
    X, y, feature_names = preprocessor.prepare_features(df_encoded)
    class_names = BINARY_CLASS_NAMES  # ["Tidak Stunting", "Stunting"]
    
    # 📊 Class Distribution (Binary)
    print("\n📊 Visualisasi: Binary Class Distribution...")
    visualizer.plot_class_imbalance(y, class_names=class_names, title_suffix="(Binary)")
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # 📊 Train/Test Split
    print("\n📊 Visualisasi: Train/Test Split...")
    visualizer.plot_train_test_split(y_train, y_test, class_names=class_names)
    
    # Store data
    data = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': feature_names,
        'df_original': df, 'df_encoded': df_encoded
    }
    
    # ================================================================
    # STEP 5: EDA
    # ================================================================
    print("\n" + "═"*70)
    print("  STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")
    print("═"*70)
    
    run_eda(df, visualizer)
    
    # ================================================================
    # STEP 6: MODEL TRAINING (4 ENSEMBLE ALGORITHMS)
    # ================================================================
    print("\n" + "═"*70)
    print("  STEP 6: MODEL TRAINING - 4 ENSEMBLE ALGORITHMS")
    print("═"*70)
    
    results = {}
    models = {}
    importances = {}
    
    for model_name in MODEL_NAMES:
        model, result, importance = train_single_model(
            model_name, trainer, X_train, y_train, X_test, y_test,
            feature_names, class_names, visualizer, tune_hyperparameters
        )
        results[model_name] = result
        models[model_name] = model
        importances[model_name] = importance
    
    # ================================================================
    # STEP 7: MODEL COMPARISON
    # ================================================================
    print("\n" + "═"*70)
    print("  STEP 7: MODEL COMPARISON")
    print("═"*70)
    
    generate_comparison_visualizations(
        results, y_test, class_names, importances, trainer, visualizer
    )
    
    # Model comparison summary
    compare_models(results)
    
    # ================================================================
    # STEP 8: SAVE RESULTS
    # ================================================================
    print("\n" + "═"*70)
    print("  STEP 8: SAVE RESULTS")
    print("═"*70)
    
    save_results(results, trainer)
    
    # Terminal summary
    print_terminal_summary(results, class_names)
    
    # Final message
    print("\n" + "="*80)
    print("  ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - Figures saved in: {FIGURES_PATH}")
    print(f"  - Results saved in: reports/")
    
    return results, models


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Binary Stunting Classification with Ensemble Learning'
    )
    parser.add_argument('--no-tuning', action='store_true',
                       help='Skip hyperparameter tuning (faster)')
    parser.add_argument('--no-inline', action='store_true',
                       help='Disable inline image display')
    
    args = parser.parse_args()
    
    results, models = main(
        tune_hyperparameters=not args.no_tuning,
        display_inline=not args.no_inline
    )
