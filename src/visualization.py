"""
Visualization Module for Stunting Classification
Creates comprehensive plots for EDA and model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.tree import plot_tree
import warnings
import time
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import FIGURES_PATH, FIGURE_DPI, FIGURE_FORMAT, TARGET_COLUMN

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Flag to enable/disable inline display
DISPLAY_INLINE = True


def display_image_inline(image_path):
    """Display image inline in terminal using imgcat."""
    if not DISPLAY_INLINE:
        return
    try:
        from imgcat import imgcat
        sys.stdout.flush()
        time.sleep(0.2)
        with open(image_path, 'rb') as f:
            imgcat(f.read())
        sys.stdout.flush()
        time.sleep(0.1)
    except ImportError:
        pass  # imgcat not installed, skip inline display
    except Exception:
        pass  # Any other error, skip silently


class Visualizer:
    """Class for creating visualizations."""
    
    def __init__(self, save_path=FIGURES_PATH, display_inline=True):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.display_inline = display_inline
        
    def save_figure(self, fig, filename):
        """Save figure to disk and optionally display inline."""
        filepath = self.save_path / f"{filename}.{FIGURE_FORMAT}"
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        print(f"\n📊 Figure saved: {filepath}")
        plt.close(fig)
        
        # Display inline in terminal
        if self.display_inline:
            display_image_inline(filepath)
        
        return filepath
    
    # ==================== PREPROCESSING VISUALIZATIONS ====================
    
    def plot_data_overview(self, df, target_col=TARGET_COLUMN):
        """Plot data overview: shape, types, missing values."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Data types distribution
        ax1 = axes[0, 0]
        dtype_counts = df.dtypes.astype(str).value_counts()
        colors = sns.color_palette("husl", len(dtype_counts))
        bars = ax1.bar(dtype_counts.index, dtype_counts.values, color=colors, edgecolor='black')
        ax1.set_title('Distribusi Tipe Data', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Tipe Data')
        ax1.set_ylabel('Jumlah Kolom')
        for bar, val in zip(bars, dtype_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', va='bottom', fontsize=11)
        
        # 2. Missing values
        ax2 = axes[0, 1]
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_cols = missing[missing > 0]
            bars = ax2.barh(missing_cols.index, missing_cols.values, color='coral', edgecolor='black')
            ax2.set_title('Missing Values per Kolom', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Jumlah Missing')
            for bar, val in zip(bars, missing_cols.values):
                ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{val} ({val/len(df)*100:.1f}%)', va='center')
        else:
            ax2.text(0.5, 0.5, '✅ Tidak ada Missing Values!', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            ax2.set_title('Missing Values', fontsize=12, fontweight='bold')
            ax2.axis('off')
        
        # 3. Target distribution
        ax3 = axes[1, 0]
        if target_col in df.columns:
            target_counts = df[target_col].value_counts()
            colors = sns.color_palette("husl", len(target_counts))
            bars = ax3.bar(target_counts.index, target_counts.values, color=colors, edgecolor='black')
            ax3.set_title(f'Distribusi Target: {target_col}', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Kelas')
            ax3.set_ylabel('Jumlah')
            ax3.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars, target_counts.values):
                pct = val / len(df) * 100
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                        f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        # 4. Dataset info summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        info_text = f"""
        📊 RINGKASAN DATASET
        
        Total Rows: {len(df):,}
        Total Columns: {len(df.columns)}
        
        Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        
        Duplicates: {df.duplicated().sum():,}
        Missing Values: {df.isnull().sum().sum():,}
        
        Numerical Columns: {len(df.select_dtypes(include=[np.number]).columns)}
        Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}
        """
        ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('DATA OVERVIEW', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, '01_data_overview')
    
    def plot_outliers(self, df, numerical_cols):
        """Plot outlier detection using boxplots and IQR."""
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6))
        
        if n_cols == 1:
            axes = [axes]
        
        outlier_info = {}
        
        for ax, col in zip(axes, numerical_cols):
            if col in df.columns:
                data = df[col].dropna()
                
                # Calculate IQR
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outlier_info[col] = len(outliers)
                
                # Boxplot
                bp = ax.boxplot(data, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][0].set_edgecolor('black')
                
                # Scatter outliers
                if len(outliers) > 0:
                    ax.scatter([1]*len(outliers), outliers, c='red', marker='x', s=50, label=f'Outliers ({len(outliers)})')
                    ax.legend(loc='upper right')
                
                ax.set_title(f'{col}\nOutliers: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)', 
                            fontsize=11, fontweight='bold')
                ax.set_ylabel(col)
                
                # Add statistics text
                stats_text = f'Min: {data.min():.2f}\nQ1: {Q1:.2f}\nMedian: {data.median():.2f}\nQ3: {Q3:.2f}\nMax: {data.max():.2f}'
                ax.text(1.15, 0.5, stats_text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('OUTLIER DETECTION (IQR Method)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, '02_outlier_detection')
    
    def plot_class_imbalance(self, y_before, y_after=None, class_names=None, title_suffix=""):
        """Plot class distribution before and optionally after balancing."""
        if y_after is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Before balancing
            ax1 = axes[0]
            unique, counts = np.unique(y_before, return_counts=True)
            colors = sns.color_palette("husl", len(unique))
            labels = [class_names[i] if class_names is not None else str(i) for i in unique]
            bars = ax1.bar(labels, counts, color=colors, edgecolor='black')
            ax1.set_title('SEBELUM Balancing', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Kelas')
            ax1.set_ylabel('Jumlah')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add imbalance ratio
            imbalance_ratio = max(counts) / min(counts)
            for bar, val in zip(bars, counts):
                pct = val / len(y_before) * 100
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
            ax1.text(0.02, 0.98, f'Imbalance Ratio: {imbalance_ratio:.2f}', transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))
            
            # After balancing
            ax2 = axes[1]
            unique, counts = np.unique(y_after, return_counts=True)
            labels = [class_names[i] if class_names is not None else str(i) for i in unique]
            bars = ax2.bar(labels, counts, color=colors, edgecolor='black')
            ax2.set_title('SESUDAH Balancing', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Kelas')
            ax2.set_ylabel('Jumlah')
            ax2.tick_params(axis='x', rotation=45)
            
            imbalance_ratio_after = max(counts) / min(counts) if min(counts) > 0 else float('inf')
            for bar, val in zip(bars, counts):
                pct = val / len(y_after) * 100
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
            ax2.text(0.02, 0.98, f'Imbalance Ratio: {imbalance_ratio_after:.2f}', transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
            
            plt.suptitle(f'CLASS IMBALANCE ANALYSIS {title_suffix}', fontsize=14, fontweight='bold')
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            unique, counts = np.unique(y_before, return_counts=True)
            colors = sns.color_palette("husl", len(unique))
            labels = [class_names[i] if class_names is not None else str(i) for i in unique]
            bars = ax.bar(labels, counts, color=colors, edgecolor='black')
            ax.set_title(f'Distribusi Kelas {title_suffix}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Kelas')
            ax.set_ylabel('Jumlah')
            ax.tick_params(axis='x', rotation=45)
            
            imbalance_ratio = max(counts) / min(counts)
            for bar, val in zip(bars, counts):
                pct = val / len(y_before) * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
            ax.text(0.02, 0.98, f'Imbalance Ratio: {imbalance_ratio:.2f}', transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))
            
            plt.suptitle(f'CLASS DISTRIBUTION {title_suffix}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = '03_class_imbalance' if y_after is None else '03_class_balance_comparison'
        return self.save_figure(fig, filename)
    
    def plot_train_test_split(self, y_train, y_test, class_names=None):
        """Plot train/test split distribution."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        colors = sns.color_palette("husl", len(np.unique(y_train)))
        
        # 1. Overall split
        ax1 = axes[0]
        sizes = [len(y_train), len(y_test)]
        labels = [f'Training\n{len(y_train):,} ({len(y_train)/(len(y_train)+len(y_test))*100:.1f}%)',
                 f'Testing\n{len(y_test):,} ({len(y_test)/(len(y_train)+len(y_test))*100:.1f}%)']
        ax1.pie(sizes, labels=labels, colors=['#3498db', '#e74c3c'], autopct='',
               explode=[0.02, 0.02], shadow=True, startangle=90)
        ax1.set_title('Train/Test Split', fontsize=12, fontweight='bold')
        
        # 2. Training set distribution
        ax2 = axes[1]
        unique, counts = np.unique(y_train, return_counts=True)
        labels = [class_names[i] if class_names is not None else str(i) for i in unique]
        bars = ax2.bar(labels, counts, color=colors, edgecolor='black')
        ax2.set_title('Training Set Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Kelas')
        ax2.set_ylabel('Jumlah')
        ax2.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f'{val}', ha='center', va='bottom', fontsize=9)
        
        # 3. Testing set distribution
        ax3 = axes[2]
        unique, counts = np.unique(y_test, return_counts=True)
        labels = [class_names[i] if class_names is not None else str(i) for i in unique]
        bars = ax3.bar(labels, counts, color=colors, edgecolor='black')
        ax3.set_title('Testing Set Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Kelas')
        ax3.set_ylabel('Jumlah')
        ax3.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{val}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('TRAIN/TEST SPLIT VISUALIZATION', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, '04_train_test_split')
    
    def plot_duplicates_analysis(self, total_before, total_after, duplicates_removed):
        """Plot duplicates removal analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Before/After comparison
        ax1 = axes[0]
        categories = ['Sebelum', 'Sesudah']
        values = [total_before, total_after]
        colors = ['#e74c3c', '#2ecc71']
        bars = ax1.bar(categories, values, color=colors, edgecolor='black')
        ax1.set_title('Jumlah Data Sebelum & Sesudah\nHapus Duplikat', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Jumlah Rows')
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. Pie chart
        ax2 = axes[1]
        sizes = [total_after, duplicates_removed]
        labels = [f'Data Unik\n{total_after:,}', f'Duplikat Dihapus\n{duplicates_removed:,}']
        colors = ['#2ecc71', '#e74c3c']
        explode = [0.02, 0.05]
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                           explode=explode, shadow=True, startangle=90)
        ax2.set_title('Proporsi Data Unik vs Duplikat', fontsize=12, fontweight='bold')
        
        plt.suptitle('DUPLICATE REMOVAL ANALYSIS', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, '02b_duplicate_analysis')
    
    # ==================== EDA VISUALIZATIONS ====================
    
    def plot_target_distribution(self, df, target_col=TARGET_COLUMN):
        """Plot distribution of target variable."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        target_counts = df[target_col].value_counts()
        colors = sns.color_palette("husl", len(target_counts))
        
        ax1 = axes[0]
        bars = ax1.bar(target_counts.index, target_counts.values, color=colors, edgecolor='black')
        ax1.set_xlabel('Status Gizi', fontsize=12)
        ax1.set_ylabel('Jumlah', fontsize=12)
        ax1.set_title('Distribusi Status Gizi Balita', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, target_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                    f'{val:,}', ha='center', va='bottom', fontsize=10)
        
        # Pie chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(
            target_counts.values, 
            labels=target_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(target_counts),
            shadow=True
        )
        ax2.set_title('Proporsi Status Gizi Balita', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return self.save_figure(fig, 'target_distribution')
    
    def plot_numerical_distributions(self, df, numerical_cols):
        """Plot distributions of numerical features."""
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(2, n_cols, figsize=(7*n_cols, 10))
        
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(numerical_cols):
            # Histogram
            axes[0, i].hist(df[col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[0, i].set_xlabel(col, fontsize=11)
            axes[0, i].set_ylabel('Frekuensi', fontsize=11)
            axes[0, i].set_title(f'Distribusi {col}', fontsize=12, fontweight='bold')
            
            # Add statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            axes[0, i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[0, i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            axes[0, i].legend()
            
            # Box plot
            sns.boxplot(data=df, y=col, ax=axes[1, i], color='lightblue')
            axes[1, i].set_ylabel(col, fontsize=11)
            axes[1, i].set_title(f'Box Plot {col}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return self.save_figure(fig, 'numerical_distributions')
    
    def plot_categorical_distributions(self, df, categorical_cols, target_col=TARGET_COLUMN):
        """Plot distributions of categorical features."""
        n_cols = len(categorical_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(8*n_cols, 6))
        
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(categorical_cols):
            cross_tab = pd.crosstab(df[col], df[target_col])
            cross_tab.plot(kind='bar', ax=axes[i], colormap='tab10', edgecolor='black')
            axes[i].set_xlabel(col, fontsize=11)
            axes[i].set_ylabel('Jumlah', fontsize=11)
            axes[i].set_title(f'Distribusi {col} berdasarkan Status Gizi', fontsize=12, fontweight='bold')
            axes[i].legend(title='Status Gizi')
            axes[i].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        return self.save_figure(fig, 'categorical_distributions')
    
    def plot_correlation_heatmap(self, df, numerical_cols):
        """Plot correlation heatmap for numerical features."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_matrix = df[numerical_cols].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            fmt='.3f', 
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            annot_kws={'size': 12}
        )
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return self.save_figure(fig, 'correlation_heatmap')
    
    def plot_feature_by_target(self, df, feature_col, target_col=TARGET_COLUMN):
        """Plot feature distribution by target class."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = df[target_col].unique()
        colors = sns.color_palette("husl", len(categories))
        
        for i, cat in enumerate(categories):
            subset = df[df[target_col] == cat][feature_col]
            ax.hist(subset, bins=40, alpha=0.5, label=cat, color=colors[i], edgecolor='black')
        
        ax.set_xlabel(feature_col, fontsize=12)
        ax.set_ylabel('Frekuensi', fontsize=12)
        ax.set_title(f'Distribusi {feature_col} per Status Gizi', fontsize=14, fontweight='bold')
        ax.legend(title='Status Gizi')
        
        plt.tight_layout()
        return self.save_figure(fig, f'feature_{feature_col.replace(" ", "_").lower()}_by_target')
    
    # ==================== MODEL EVALUATION VISUALIZATIONS ====================
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, model_name="Model"):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize for percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotations with both counts and percentages
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)'
        
        sns.heatmap(
            cm, 
            annot=annotations,
            fmt='',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            annot_kws={'size': 10}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return self.save_figure(fig, f'confusion_matrix_{model_name.lower().replace(" ", "_")}')
    
    def plot_roc_curves(self, results_dict, y_test, class_names):
        """Plot ROC curves for all models."""
        n_classes = len(class_names)
        
        if n_classes == 2:
            # Binary classification
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = sns.color_palette("husl", len(results_dict))
            
            for (model_name, results), color in zip(results_dict.items(), colors):
                if results['y_proba'] is not None:
                    fpr, tpr, _ = roc_curve(y_test, results['y_proba'][:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=color, lw=2, 
                           label=f'{model_name} (AUC = {roc_auc:.4f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
        else:
            # Multi-class: One plot per model
            fig, axes = plt.subplots(1, len(results_dict), figsize=(8*len(results_dict), 6))
            
            if len(results_dict) == 1:
                axes = [axes]
            
            for ax, (model_name, results) in zip(axes, results_dict.items()):
                if results['y_proba'] is not None:
                    colors = sns.color_palette("husl", n_classes)
                    
                    for i, (class_name, color) in enumerate(zip(class_names, colors)):
                        # One-vs-Rest ROC
                        y_true_binary = (y_test == i).astype(int)
                        fpr, tpr, _ = roc_curve(y_true_binary, results['y_proba'][:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, color=color, lw=2,
                               label=f'{class_name} (AUC = {roc_auc:.4f})')
                    
                    ax.plot([0, 1], [0, 1], 'k--', lw=2)
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate', fontsize=11)
                    ax.set_ylabel('True Positive Rate', fontsize=11)
                    ax.set_title(f'ROC Curve - {model_name}', fontsize=12, fontweight='bold')
                    ax.legend(loc='lower right', fontsize=9)
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.save_figure(fig, 'roc_curves')
    
    def plot_metrics_comparison(self, results_dict):
        """Plot bar chart comparing model metrics."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(results_dict.keys())
        
        x = np.arange(len(metrics))
        width = 0.35
        
        colors = sns.color_palette("husl", len(model_names))
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            values = [results_dict[model_name][m] for m in metrics]
            offset = width * (i - len(model_names)/2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=model_name, color=color, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.set_ylim(0, 1.15)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        return self.save_figure(fig, 'metrics_comparison')
    
    def plot_feature_importance(self, importance_dict, model_name="Model"):
        """Plot feature importance."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)
        
        colors = sns.color_palette("viridis", len(features))
        
        bars = ax.barh(range(len(features)), importances, color=colors, edgecolor='black')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, importances):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=10)
        
        ax.invert_yaxis()
        plt.tight_layout()
        return self.save_figure(fig, f'feature_importance_{model_name.lower().replace(" ", "_")}')
    
    def plot_decision_tree(self, model, feature_names, class_names, model_name="Decision Tree"):
        """Plot decision tree structure."""
        fig, ax = plt.subplots(figsize=(25, 15))
        
        plot_tree(
            model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax,
            max_depth=4  # Limit depth for readability
        )
        ax.set_title(f'Decision Tree Structure - {model_name} (Max Depth: 4)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return self.save_figure(fig, f'decision_tree_{model_name.lower().replace(" ", "_")}')
    
    def plot_cv_results(self, cv_results_dict):
        """Plot cross-validation results comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        colors = sns.color_palette("husl", len(cv_results_dict))
        
        for ax, metric in zip(axes, metrics):
            data = []
            labels = []
            
            for model_name, results in cv_results_dict.items():
                if metric in results:
                    data.append(results[metric])
                    labels.append(model_name)
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()} - Cross Validation', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        return self.save_figure(fig, 'cv_results_comparison')
    
    def plot_model_comparison_summary(self, results_dict):
        """Plot comprehensive model comparison summary."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        model_names = list(results_dict.keys())
        colors = ['#3498db', '#e74c3c']  # Blue for DT, Red for RF
        
        # 1. Bar chart - All metrics comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            values = [results_dict[model_name].get(m, 0) or 0 for m in metrics]
            offset = width * (i - 0.5)
            bars = ax1.bar(x + offset, values, width, label=model_name, color=color, edgecolor='black')
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Perbandingan Semua Metrik', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax1.legend(loc='lower right')
        ax1.set_ylim(0.95, 1.02)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # 2. Radar/Spider chart (top right)
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model_name, color in zip(model_names, colors):
            values = [results_dict[model_name].get(m, 0) or 0 for m in metrics]
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax2.fill(angles, values, alpha=0.25, color=color)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metric_labels, fontsize=9)
        ax2.set_ylim(0.95, 1.0)
        ax2.set_title('Radar Chart Perbandingan', fontsize=12, fontweight='bold', y=1.1)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 3. Accuracy comparison with percentage (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        accuracies = [results_dict[m]['accuracy'] * 100 for m in model_names]
        bars = ax3.barh(model_names, accuracies, color=colors, edgecolor='black', height=0.5)
        ax3.set_xlim(95, 100)
        ax3.set_xlabel('Accuracy (%)', fontsize=11)
        ax3.set_title('Perbandingan Akurasi', fontsize=12, fontweight='bold')
        
        for bar, acc in zip(bars, accuracies):
            ax3.text(acc + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{acc:.2f}%', va='center', fontsize=11, fontweight='bold')
        
        # Add winner indicator
        winner_idx = np.argmax(accuracies)
        ax3.text(accuracies[winner_idx] - 2, bars[winner_idx].get_y() + bars[winner_idx].get_height()/2,
                '★ BEST', va='center', ha='right', fontsize=10, color='gold', fontweight='bold')
        ax3.grid(True, axis='x', alpha=0.3)
        
        # 4. Training time comparison (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        train_times = [results_dict[m].get('training_time', 0) for m in model_names]
        bars = ax4.bar(model_names, train_times, color=colors, edgecolor='black')
        ax4.set_ylabel('Time (seconds)', fontsize=11)
        ax4.set_title('Waktu Training', fontsize=12, fontweight='bold')
        
        for bar, t in zip(bars, train_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{t:.3f}s', ha='center', va='bottom', fontsize=10)
        ax4.grid(True, axis='y', alpha=0.3)
        
        # 5. Difference analysis (bottom - full width)
        ax5 = fig.add_subplot(gs[2, :])
        
        if len(model_names) >= 2:
            diff_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            diff_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            
            # Calculate differences (RF - DT)
            differences = []
            for m in diff_metrics:
                val1 = results_dict[model_names[0]].get(m, 0) or 0
                val2 = results_dict[model_names[1]].get(m, 0) or 0
                differences.append((val2 - val1) * 100)  # Convert to percentage points
            
            x = np.arange(len(diff_metrics))
            bar_colors = ['green' if d > 0 else 'red' for d in differences]
            bars = ax5.bar(x, differences, color=bar_colors, edgecolor='black', alpha=0.7)
            
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax5.set_ylabel('Difference (percentage points)', fontsize=11)
            ax5.set_title(f'Selisih Performa: {model_names[1]} vs {model_names[0]}\n(Positif = {model_names[1]} lebih baik)', 
                         fontsize=12, fontweight='bold')
            ax5.set_xticks(x)
            ax5.set_xticklabels(diff_labels)
            
            for bar, diff in zip(bars, differences):
                label = f'+{diff:.2f}' if diff > 0 else f'{diff:.2f}'
                y_pos = diff + 0.02 if diff > 0 else diff - 0.05
                ax5.text(bar.get_x() + bar.get_width()/2, y_pos,
                        label, ha='center', va='bottom' if diff > 0 else 'top', fontsize=10)
            ax5.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle('KOMPARASI MODEL: Decision Tree C4.5 vs Random Forest', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return self.save_figure(fig, 'model_comparison_summary')
    
    def plot_confusion_matrix_comparison(self, results_dict, y_test, class_names):
        """Plot side-by-side confusion matrices for comparison."""
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 7))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, results) in zip(axes, results_dict.items()):
            cm = confusion_matrix(y_test, results['y_pred'])
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create annotations
            annotations = np.empty_like(cm).astype(str)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)'
            
            sns.heatmap(
                cm, 
                annot=annotations,
                fmt='',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                annot_kws={'size': 9}
            )
            
            # Calculate accuracy
            acc = np.trace(cm) / cm.sum() * 100
            
            ax.set_xlabel('Predicted Label', fontsize=11)
            ax.set_ylabel('True Label', fontsize=11)
            ax.set_title(f'{model_name}\nAccuracy: {acc:.2f}%', fontsize=12, fontweight='bold')
        
        plt.suptitle('Perbandingan Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, 'confusion_matrix_comparison')
    
    def plot_per_class_comparison(self, results_dict, y_test, class_names):
        """Plot per-class metrics comparison between models."""
        from sklearn.metrics import precision_recall_fscore_support
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        model_names = list(results_dict.keys())
        colors = ['#3498db', '#e74c3c']
        
        x = np.arange(len(class_names))
        width = 0.35
        
        metrics_data = {}
        for model_name, results in results_dict.items():
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, results['y_pred'], average=None
            )
            metrics_data[model_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        metric_titles = ['Precision per Kelas', 'Recall per Kelas', 'F1-Score per Kelas']
        metric_keys = ['precision', 'recall', 'f1']
        
        for ax, metric_key, title in zip(axes, metric_keys, metric_titles):
            for i, (model_name, color) in enumerate(zip(model_names, colors)):
                values = metrics_data[model_name][metric_key]
                offset = width * (i - 0.5)
                bars = ax.bar(x + offset, values, width, label=model_name, color=color, edgecolor='black')
                
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
            
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.legend(loc='lower right')
            ax.set_ylim(0.9, 1.02)
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle('Perbandingan Performa per Kelas', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, 'per_class_comparison')
    
    def plot_feature_importance_comparison(self, dt_importance, rf_importance):
        """Plot side-by-side feature importance comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        importances = [dt_importance, rf_importance]
        titles = ['Decision Tree C4.5', 'Random Forest']
        colors = ['#3498db', '#e74c3c']
        
        for ax, imp_dict, title, color in zip(axes, importances, titles, colors):
            sorted_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
            features, values = zip(*sorted_features)
            
            bars = ax.barh(range(len(features)), values, color=color, edgecolor='black', alpha=0.8)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title(f'Feature Importance - {title}', fontsize=12, fontweight='bold')
            
            for bar, val in zip(bars, values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.4f} ({val*100:.1f}%)', va='center', fontsize=10)
            
            ax.invert_yaxis()
            ax.set_xlim(0, max(values) * 1.3)
        
        plt.suptitle('Perbandingan Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, 'feature_importance_comparison')
    
    # ==================== BINARY CLASSIFICATION VISUALIZATIONS ====================
    
    def plot_roc_curves_binary(self, results_dict, y_test):
        """Plot ROC curves for binary classification with all models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = sns.color_palette("husl", len(results_dict))
        
        for (model_name, results), color in zip(results_dict.items(), colors):
            if results.get('y_proba') is not None:
                fpr, tpr, _ = roc_curve(y_test, results['y_proba'][:, 1])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=2.5, 
                       label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve Comparison - Binary Classification\n(Stunting vs Tidak Stunting)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add annotation for AUC interpretation
        ax.text(0.6, 0.2, 'AUC > 0.9 = Excellent\nAUC 0.8-0.9 = Good\nAUC 0.7-0.8 = Fair', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        return self.save_figure(fig, 'roc_curves_binary')
    
    def plot_precision_recall_curves(self, results_dict, y_test):
        """Plot Precision-Recall curves for binary classification."""
        from sklearn.metrics import average_precision_score
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = sns.color_palette("husl", len(results_dict))
        
        for (model_name, results), color in zip(results_dict.items(), colors):
            if results.get('y_proba') is not None:
                precision, recall, _ = precision_recall_curve(y_test, results['y_proba'][:, 1])
                ap = average_precision_score(y_test, results['y_proba'][:, 1])
                ax.plot(recall, precision, color=color, lw=2.5,
                       label=f'{model_name} (AP = {ap:.4f})')
        
        # Baseline (random classifier)
        baseline = y_test.sum() / len(y_test)
        ax.axhline(y=baseline, color='k', linestyle='--', lw=2, label=f'Baseline (AP = {baseline:.4f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve Comparison\n(Stunting Detection)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.save_figure(fig, 'precision_recall_curves')
    
    def plot_feature_importance_comparison_multi(self, importances_dict):
        """Plot feature importance comparison for multiple models (4 ensemble algorithms)."""
        n_models = len(importances_dict)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']  # RF, XGB, LGBM, CB
        
        for ax, (model_name, imp_dict), color in zip(axes, importances_dict.items(), colors):
            sorted_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
            features, values = zip(*sorted_features)
            
            bars = ax.barh(range(len(features)), values, color=color, edgecolor='black', alpha=0.8)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title(f'Feature Importance - {model_name}', fontsize=12, fontweight='bold')
            
            for bar, val in zip(bars, values):
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{val:.4f}', va='center', fontsize=9)
            
            ax.invert_yaxis()
            ax.set_xlim(0, max(values) * 1.25)
        
        plt.suptitle('Perbandingan Feature Importance - 4 Ensemble Algorithms', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, 'feature_importance_comparison_multi')
    
    def plot_model_comparison_summary(self, results_dict):
        """Plot comprehensive model comparison summary for 4 ensemble models."""
        fig = plt.figure(figsize=(18, 14))
        
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        model_names = list(results_dict.keys())
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']  # RF, XGB, LGBM, CB
        
        # 1. Bar chart - All metrics comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(metrics))
        width = 0.2
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            values = [results_dict[model_name].get(m, 0) or 0 for m in metrics]
            offset = width * (i - 1.5)
            bars = ax1.bar(x + offset, values, width, label=model_name, color=color, edgecolor='black')
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
        
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Perbandingan Semua Metrik', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.set_ylim(0, 1.15)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # 2. Radar/Spider chart (top right)
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for model_name, color in zip(model_names, colors):
            values = [results_dict[model_name].get(m, 0) or 0 for m in metrics]
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax2.fill(angles, values, alpha=0.15, color=color)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metric_labels, fontsize=9)
        ax2.set_ylim(0, 1.0)
        ax2.set_title('Radar Chart Perbandingan', fontsize=12, fontweight='bold', y=1.1)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=9)
        
        # 3. Accuracy comparison with percentage (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        accuracies = [results_dict[m]['accuracy'] * 100 for m in model_names]
        bars = ax3.barh(model_names, accuracies, color=colors, edgecolor='black', height=0.6)
        ax3.set_xlim(min(accuracies) - 5, 100)
        ax3.set_xlabel('Accuracy (%)', fontsize=11)
        ax3.set_title('Perbandingan Akurasi', fontsize=12, fontweight='bold')
        
        for bar, acc in zip(bars, accuracies):
            ax3.text(acc + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{acc:.2f}%', va='center', fontsize=10, fontweight='bold')
        
        winner_idx = np.argmax(accuracies)
        ax3.text(accuracies[winner_idx] - 3, bars[winner_idx].get_y() + bars[winner_idx].get_height()/2,
                '★', va='center', ha='right', fontsize=14, color='gold')
        ax3.grid(True, axis='x', alpha=0.3)
        
        # 4. F1-Score comparison (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        f1_scores = [results_dict[m].get('f1', 0) * 100 for m in model_names]
        bars = ax4.barh(model_names, f1_scores, color=colors, edgecolor='black', height=0.6)
        ax4.set_xlim(min(f1_scores) - 5, 100)
        ax4.set_xlabel('F1-Score (%)', fontsize=11)
        ax4.set_title('Perbandingan F1-Score', fontsize=12, fontweight='bold')
        
        for bar, f1 in zip(bars, f1_scores):
            ax4.text(f1 + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{f1:.2f}%', va='center', fontsize=10, fontweight='bold')
        
        winner_idx = np.argmax(f1_scores)
        ax4.text(f1_scores[winner_idx] - 3, bars[winner_idx].get_y() + bars[winner_idx].get_height()/2,
                '★', va='center', ha='right', fontsize=14, color='gold')
        ax4.grid(True, axis='x', alpha=0.3)
        
        # 5. Training time comparison (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        train_times = [results_dict[m].get('training_time', 0) for m in model_names]
        bars = ax5.bar(model_names, train_times, color=colors, edgecolor='black')
        ax5.set_ylabel('Time (seconds)', fontsize=11)
        ax5.set_title('Waktu Training', fontsize=12, fontweight='bold')
        ax5.tick_params(axis='x', rotation=15)
        
        for bar, t in zip(bars, train_times):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{t:.3f}s', ha='center', va='bottom', fontsize=9)
        ax5.grid(True, axis='y', alpha=0.3)
        
        # 6. Winner summary (bottom right)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        # Find best model based on F1-Score
        best_idx = np.argmax(f1_scores)
        best_model = model_names[best_idx]
        best_acc = accuracies[best_idx]
        best_f1 = f1_scores[best_idx]
        best_auc = results_dict[best_model].get('roc_auc', 0) * 100
        
        summary_text = f"""
        🏆 HASIL KOMPARASI - BEST MODEL 🏆
        
        Model Terbaik: {best_model}
        
        Metrik:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        • Accuracy:  {best_acc:.2f}%
        • F1-Score:  {best_f1:.2f}%
        • ROC-AUC:   {best_auc:.2f}%
        
        Ranking (berdasarkan F1-Score):
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        sorted_models = sorted(zip(model_names, f1_scores), key=lambda x: x[1], reverse=True)
        medals = ['🥇', '🥈', '🥉', '4️⃣']
        for i, (name, f1) in enumerate(sorted_models):
            summary_text += f"        {medals[i]} {name}: {f1:.2f}%\n"
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gold', linewidth=2))
        
        plt.suptitle('KOMPARASI MODEL ENSEMBLE LEARNING\n(Random Forest, XGBoost, LightGBM, CatBoost)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return self.save_figure(fig, 'model_comparison_summary')


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
