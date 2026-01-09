"""
Configuration file for Stunting Classification Project
Skripsi: Analisis Komparasi Kinerja Algoritma Ensemble Learning
         (Random Forest, XGBoost, LightGBM, dan CatBoost)
         untuk Klasifikasi Biner Status Stunting
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"

# Dataset Configuration
DATASET_NAME = "rendiputra/stunting-balita-detection-121k-rows"
RAW_DATA_FILE = DATA_RAW_PATH / "stunting_data.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_PATH / "stunting_processed.csv"

# Feature Configuration
TARGET_COLUMN = "Status Gizi"  # Original target variable
BINARY_TARGET_COLUMN = "Stunting_Status"  # Binary target (0: Tidak Stunting, 1: Stunting)
CATEGORICAL_FEATURES = ["Jenis Kelamin"]
NUMERICAL_FEATURES = ["Umur (bulan)", "Tinggi Badan (cm)"]

# Binary Classification Mapping
# 0 = Tidak Stunting (normal, tinggi)
# 1 = Stunting (stunted, severely stunted)
BINARY_CLASS_MAPPING = {
    "normal": 0,
    "tinggi": 0,
    "stunted": 1,
    "severely stunted": 1
}
BINARY_CLASS_NAMES = ["Tidak Stunting", "Stunting"]

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10  # For cross-validation

# ==================== ENSEMBLE MODEL PARAMETERS ====================

# Random Forest Parameters
RF_PARAMS = {
    "criterion": "gini",
    "random_state": RANDOM_STATE,
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
}

# XGBoost Parameters
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbosity": 0,
}

# LightGBM Parameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "random_state": RANDOM_STATE,
    "n_estimators": 100,
    "max_depth": -1,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbose": -1,
}

# CatBoost Parameters
CB_PARAMS = {
    "loss_function": "Logloss",
    "random_state": RANDOM_STATE,
    "iterations": 100,
    "depth": 6,
    "learning_rate": 0.1,
    "verbose": False,
}

# ==================== HYPERPARAMETER TUNING GRIDS ====================

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9],
}

LGBM_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, -1],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "num_leaves": [20, 31, 50],
}

CB_PARAM_GRID = {
    "iterations": [100, 200, 300],
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
}

# Evaluation Metrics for Binary Classification
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Visualization Settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Model Names
MODEL_NAMES = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
