"""
Model Training Module for Binary Stunting Classification
Skripsi: Analisis Performa Algoritma Machine Learning
         untuk Klasifikasi Status Stunting pada Balita

Implements machine learning algorithms used in this study:
    - Decision Tree (baseline non-ensemble)
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
"""

from __future__ import annotations

import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    RANDOM_STATE, CV_FOLDS, MODELS_PATH,
    DT_PARAMS, RF_PARAMS, XGB_PARAMS, LGBM_PARAMS, CB_PARAMS,
    DT_PARAM_GRID, RF_PARAM_GRID, XGB_PARAM_GRID, LGBM_PARAM_GRID, CB_PARAM_GRID,
    BINARY_CLASS_NAMES,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

TUNING_CV_FOLDS: int = 5  # used during GridSearch / manual CatBoost tuning


class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    Supports: Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost.
    """
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.cv_results = {}
        self.training_times = {}
        
    def create_decision_tree(
        self, params: Optional[Mapping[str, Any]] = None
    ) -> DecisionTreeClassifier:
        """Create Decision Tree classifier (baseline non-ensemble)."""
        if params is None:
            params = DT_PARAMS
        return DecisionTreeClassifier(**params)

    def create_random_forest(
        self, params: Optional[Mapping[str, Any]] = None
    ) -> RandomForestClassifier:
        """Create Random Forest classifier."""
        if params is None:
            params = RF_PARAMS
        return RandomForestClassifier(**params)

    def create_xgboost(
        self, params: Optional[Mapping[str, Any]] = None
    ) -> XGBClassifier:
        """Create XGBoost classifier."""
        if params is None:
            params = XGB_PARAMS
        return XGBClassifier(**params)

    def create_lightgbm(
        self, params: Optional[Mapping[str, Any]] = None
    ) -> LGBMClassifier:
        """Create LightGBM classifier."""
        if params is None:
            params = LGBM_PARAMS
        return LGBMClassifier(**params)

    def create_catboost(
        self, params: Optional[Mapping[str, Any]] = None
    ) -> CatBoostClassifier:
        """Create CatBoost classifier."""
        if params is None:
            params = CB_PARAMS
        return CatBoostClassifier(**params)

    def create_model(self, model_name: str) -> BaseEstimator:
        """Create model by name."""
        model_creators = {
            'Decision Tree': self.create_decision_tree,
            'Random Forest': self.create_random_forest,
            'XGBoost': self.create_xgboost,
            'LightGBM': self.create_lightgbm,
            'CatBoost': self.create_catboost,
        }
        if model_name not in model_creators:
            raise ValueError(f"Unknown model: {model_name}")
        return model_creators[model_name]()

    def get_param_grid(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter grid for model."""
        param_grids: Dict[str, Dict[str, Any]] = {
            'Decision Tree': DT_PARAM_GRID,
            'Random Forest': RF_PARAM_GRID,
            'XGBoost': XGB_PARAM_GRID,
            'LightGBM': LGBM_PARAM_GRID,
            'CatBoost': CB_PARAM_GRID,
        }
        return param_grids.get(model_name, {})
    
    def train_model(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str = "Model",
    ) -> BaseEstimator:
        """Train a model and record training time (wall-clock, monotonic)."""
        logger.info("Training %s...", model_name)
        start_time = time.perf_counter()

        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time
        self.training_times[model_name] = training_time
        logger.info("Training completed in %.4f seconds", training_time)

        return model

    def _clone_for_fold(
        self, model: BaseEstimator, model_name: str
    ) -> BaseEstimator:
        """Create a fresh copy of `model` for a single CV fold.

        For CatBoost, sklearn.base.clone() is not reliable; we instantiate
        a new CatBoostClassifier with the same parameters as the (already
        tuned) `model` so that cross-validation reflects the tuned
        configuration rather than the default one.
        """
        if model_name == 'CatBoost':
            params = model.get_params() if hasattr(model, 'get_params') else CB_PARAMS
            # CatBoost returns some parameters that cannot be passed back in
            # (e.g. "simple_ctr"); keep only the keys present in CB_PARAMS
            # plus any that the user has tuned explicitly.
            safe_params = {**CB_PARAMS, **{k: v for k, v in params.items() if k in CB_PARAMS or k in CB_PARAM_GRID}}
            return CatBoostClassifier(**safe_params)
        return clone(model)

    def cross_validate(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "Model",
        cv: int = CV_FOLDS,
    ) -> Dict[str, np.ndarray]:
        """Perform k-fold cross-validation (manual for CatBoost compatibility)."""
        logger.info("Performing %d-fold Cross-Validation for %s...", cv, model_name)

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for train_idx, val_idx in cv_strategy.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_model = self._clone_for_fold(model, model_name)
            fold_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = fold_model.predict(X_val_fold)

            accuracy_scores.append(accuracy_score(y_val_fold, y_pred_fold))
            precision_scores.append(precision_score(y_val_fold, y_pred_fold, average='weighted'))
            recall_scores.append(recall_score(y_val_fold, y_pred_fold, average='weighted'))
            f1_scores.append(f1_score(y_val_fold, y_pred_fold, average='weighted'))
        
        metrics = {
            'accuracy': np.array(accuracy_scores),
            'precision_weighted': np.array(precision_scores),
            'recall_weighted': np.array(recall_scores),
            'f1_weighted': np.array(f1_scores),
        }
        
        self.cv_results[model_name] = metrics
        
        logger.info("Cross-Validation Results for %s:", model_name)
        for metric_name, scores in metrics.items():
            logger.info(
                "  %s: Mean=%.4f (+/- %.4f)  Min=%.4f  Max=%.4f",
                metric_name,
                scores.mean(),
                scores.std() * 2,
                scores.min(),
                scores.max(),
            )

        return metrics
    
    def hyperparameter_tuning(
        self,
        model: BaseEstimator,
        param_grid: Mapping[str, Sequence[Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str = "Model",
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Perform hyperparameter tuning (manual for CatBoost compatibility)."""
        logger.info("Performing Hyperparameter Tuning for %s...", model_name)
        logger.info("Parameter grid: %s", param_grid)

        # For CatBoost, use manual grid search due to sklearn compatibility issues
        if model_name == 'CatBoost':
            return self._manual_grid_search_catboost(param_grid, X_train, y_train, model_name)

        cv_strategy = StratifiedKFold(
            n_splits=TUNING_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv_strategy,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        start_time = time.perf_counter()
        grid_search.fit(X_train, y_train)
        tuning_time = time.perf_counter() - start_time

        self.best_params[model_name] = dict(grid_search.best_params_)

        logger.info("Tuning completed in %.2f seconds", tuning_time)
        logger.info("Best parameters: %s", grid_search.best_params_)
        logger.info("Best CV score: %.4f", grid_search.best_score_)

        return grid_search.best_estimator_, dict(grid_search.best_params_)
    
    def _manual_grid_search_catboost(
        self,
        param_grid: Mapping[str, Sequence[Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Manual grid search for CatBoost due to sklearn compatibility issues."""
        cv_strategy = StratifiedKFold(
            n_splits=TUNING_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        logger.info(
            "Testing %d parameter combinations for %s",
            len(param_combinations),
            model_name,
        )

        best_score = -1.0
        best_params: Dict[str, Any] = {}
        best_model: Optional[BaseEstimator] = None

        start_time = time.perf_counter()

        progress = tqdm(
            param_combinations,
            desc=f"CatBoost grid search",
            total=len(param_combinations),
            unit="combo",
        )
        for params_tuple in progress:
            params = dict(zip(param_names, params_tuple))
            full_params = {**CB_PARAMS, **params}
            
            fold_scores = []
            
            for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                model = CatBoostClassifier(**full_params)
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                
                score = f1_score(y_fold_val, y_pred, average='weighted')
                fold_scores.append(score)
            
            mean_score = float(np.mean(fold_scores))
            progress.set_postfix(best=f"{best_score:.4f}", current=f"{mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_model = CatBoostClassifier(**{**CB_PARAMS, **best_params})

        tuning_time = time.perf_counter() - start_time

        self.best_params[model_name] = dict(best_params)

        logger.info("Tuning completed in %.2f seconds", tuning_time)
        logger.info("Best parameters: %s", best_params)
        logger.info("Best CV score: %.4f", best_score)

        # Fit best model on full training data
        assert best_model is not None, "CatBoost grid search yielded no candidate"
        best_model.fit(X_train, y_train)

        return best_model, dict(best_params)
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model", class_names=None):
        """Comprehensive model evaluation."""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS: {model_name}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # ROC-AUC (multi-class)
        if y_proba is not None:
            if len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = None
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"\n1. Overall Metrics:")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        if roc_auc is not None:
            print(f"   ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
        
        print(f"\n2. Per-Class Metrics:")
        if class_names is not None:
            for i, class_name in enumerate(class_names):
                print(f"   {class_name}:")
                print(f"      Precision: {precision_per_class[i]:.4f}")
                print(f"      Recall:    {recall_per_class[i]:.4f}")
                print(f"      F1-Score:  {f1_per_class[i]:.4f}")
        
        print(f"\n3. Confusion Matrix:")
        print(cm)
        
        print(f"\n4. Classification Report:")
        if class_names is not None:
            print(classification_report(y_test, y_pred, target_names=class_names))
        else:
            print(classification_report(y_test, y_pred))
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'training_time': self.training_times.get(model_name, None)
        }
        
        return results
    
    def get_feature_importance(self, model, feature_names, model_name="Model"):
        """Get feature importance from tree-based models."""
        print(f"\n{'='*60}")
        print(f"FEATURE IMPORTANCE: {model_name}")
        print(f"{'='*60}")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature ranking:")
        for i, idx in enumerate(indices):
            print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        return dict(zip(feature_names, importances))
    
    def save_model(self, model, model_name):
        """Save model to disk (creates MODELS_PATH if needed)."""
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        filepath = MODELS_PATH / f"{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, model_name):
        """Load model from disk."""
        filepath = MODELS_PATH / f"{model_name.lower().replace(' ', '_')}.joblib"
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model


def compare_models(results_dict):
    """Compare multiple models side by side."""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    # Create comparison dataframe
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'training_time']
    comparison_data = {}
    
    for model_name, results in results_dict.items():
        comparison_data[model_name] = {
            metric: results.get(metric, None) for metric in metrics
        }
    
    df_comparison = pd.DataFrame(comparison_data).T
    
    print("\nMetrics Comparison:")
    print("-" * 80)
    print(df_comparison.to_string())
    
    # Determine best model for each metric
    print("\n\nBest Model per Metric:")
    print("-" * 80)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        values = {name: results.get(metric, 0) or 0 for name, results in results_dict.items()}
        best_model = max(values, key=values.get)
        best_value = values[best_model]
        print(f"   {metric.upper():12}: {best_model} ({best_value:.4f})")
    
    # Determine overall winner (based on F1-score)
    f1_scores = {name: results.get('f1', 0) for name, results in results_dict.items()}
    overall_winner = max(f1_scores, key=f1_scores.get)
    
    print(f"\n{'='*80}")
    print(f"OVERALL BEST MODEL (based on F1-Score): {overall_winner}")
    print(f"{'='*80}")
    
    return df_comparison


if __name__ == "__main__":
    print("Model module loaded successfully!")
    print("Available models: Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost")
