"""
Data Preprocessing Module for Binary Stunting Classification
Handles data loading, cleaning, encoding, and feature engineering

Binary Classification:
    - 0 = Tidak Stunting (normal, tinggi)
    - 1 = Stunting (stunted, severely stunted)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    TARGET_COLUMN, BINARY_TARGET_COLUMN, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    RANDOM_STATE, TEST_SIZE, DATA_RAW_PATH, DATA_PROCESSED_PATH,
    BINARY_CLASS_MAPPING, BINARY_CLASS_NAMES
)


class DataPreprocessor:
    """
    Class for preprocessing stunting dataset for binary classification.
    
    Binary Classification:
        - 0 = Tidak Stunting (normal, tinggi)
        - 1 = Stunting (stunted, severely stunted)
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_encoder = None  # Not needed for binary, using direct mapping
        self.feature_names = None
        self.class_names = BINARY_CLASS_NAMES  # ["Tidak Stunting", "Stunting"]
        
    def load_data(self, filepath):
        """Load dataset from CSV file."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def explore_data(self, df):
        """Generate basic statistics and info about the dataset."""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        print("\n1. Dataset Shape:")
        print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        print("\n2. Column Names:")
        for col in df.columns:
            print(f"   - {col}")
        
        print("\n3. Data Types:")
        print(df.dtypes.to_string())
        
        print("\n4. Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("   No missing values found!")
        else:
            print(missing[missing > 0].to_string())
        
        print("\n5. Basic Statistics:")
        print(df.describe().to_string())
        
        print("\n6. Target Variable Distribution:")
        if TARGET_COLUMN in df.columns:
            target_dist = df[TARGET_COLUMN].value_counts()
            print(target_dist.to_string())
            print(f"\n   Percentage:")
            print((target_dist / len(df) * 100).round(2).to_string())
        
        return df
    
    def clean_data(self, df):
        """Clean dataset: handle missing values, duplicates, outliers."""
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)
        
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        print(f"1. Duplicates removed: {duplicates_removed}")
        
        # Handle missing values (avoid chained-inplace deprecation in pandas 3.0)
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            # For numerical: fill with median
            for col in NUMERICAL_FEATURES:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())

            # For categorical: fill with mode
            for col in CATEGORICAL_FEATURES:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].mode()[0])

            print(f"2. Missing values handled: {missing_before}")
        else:
            print("2. No missing values to handle")
        
        # Remove rows with missing target
        if TARGET_COLUMN in df.columns:
            df = df.dropna(subset=[TARGET_COLUMN])
        
        print(f"3. Final dataset size: {len(df)} rows")
        
        return df
    
    def encode_features(self, df):
        """Encode categorical features and create binary target."""
        print("\n" + "="*60)
        print("FEATURE ENCODING (BINARY CLASSIFICATION)")
        print("="*60)
        
        df_encoded = df.copy()
        
        # Encode categorical features
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                print(f"Encoded '{col}':")
                for i, label in enumerate(self.label_encoders[col].classes_):
                    print(f"   {label} -> {i}")
        
        # Create BINARY target variable
        if TARGET_COLUMN in df_encoded.columns:
            print(f"\n{'='*40}")
            print("BINARY TARGET MAPPING")
            print(f"{'='*40}")
            print("\nOriginal -> Binary:")
            print("  normal         -> 0 (Tidak Stunting)")
            print("  tinggi         -> 0 (Tidak Stunting)")
            print("  stunted        -> 1 (Stunting)")
            print("  severely stunted -> 1 (Stunting)")
            
            # Apply binary mapping
            df_encoded[BINARY_TARGET_COLUMN] = df_encoded[TARGET_COLUMN].map(BINARY_CLASS_MAPPING)
            
            # Show distribution
            print(f"\nBinary Target Distribution:")
            binary_dist = df_encoded[BINARY_TARGET_COLUMN].value_counts().sort_index()
            for idx, count in binary_dist.items():
                class_name = BINARY_CLASS_NAMES[idx]
                pct = count / len(df_encoded) * 100
                print(f"   {idx} ({class_name}): {count:,} ({pct:.2f}%)")
        
        return df_encoded
    
    def prepare_features(self, df):
        """Prepare feature matrix X and binary target vector y."""
        print("\n" + "="*60)
        print("FEATURE PREPARATION (BINARY)")
        print("="*60)
        
        # Define feature columns
        feature_cols = []
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                feature_cols.append(col)
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                feature_cols.append(col)
        
        self.feature_names = feature_cols
        print(f"Features used: {feature_cols}")
        
        X = df[feature_cols].values
        y = df[BINARY_TARGET_COLUMN].values  # Binary target
        
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Binary Classes: {BINARY_CLASS_NAMES}")
        print(f"  0 = {BINARY_CLASS_NAMES[0]}")
        print(f"  1 = {BINARY_CLASS_NAMES[1]}")
        
        return X, y, feature_cols
    
    def split_data(self, X, y, test_size=TEST_SIZE, stratify=True):
        """Split data into training and testing sets."""
        print("\n" + "="*60)
        print("DATA SPLITTING")
        print("="*60)
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=stratify_param
        )
        
        print(f"Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"Testing set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
        
        # Check class distribution
        print("\nClass distribution in training set:")
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"   Class {u} ({BINARY_CLASS_NAMES[u]}): {c} ({c/len(y_train)*100:.2f}%)")
        
        print("\nClass distribution in testing set:")
        unique, counts = np.unique(y_test, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"   Class {u} ({BINARY_CLASS_NAMES[u]}): {c} ({c/len(y_test)*100:.2f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def full_pipeline(self, filepath):
        """Run complete preprocessing pipeline."""
        print("\n" + "="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        # 1. Load data
        df = self.load_data(filepath)
        
        # 2. Explore data
        df = self.explore_data(df)
        
        # 3. Clean data
        df = self.clean_data(df)
        
        # 4. Encode features
        df_encoded = self.encode_features(df)
        
        # 5. Prepare features
        X, y, feature_names = self.prepare_features(df_encoded)
        
        # 6. Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED")
        print("="*60)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'target_encoder': self.target_encoder,
            'df_original': df,
            'df_encoded': df_encoded
        }


def get_class_names(preprocessor):
    """Get class names from preprocessor."""
    return preprocessor.target_encoder.classes_


if __name__ == "__main__":
    # Test preprocessing
    from config import RAW_DATA_FILE
    
    preprocessor = DataPreprocessor()
    data = preprocessor.full_pipeline(RAW_DATA_FILE)
    
    print("\nPreprocessing test completed!")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Testing samples: {len(data['X_test'])}")
