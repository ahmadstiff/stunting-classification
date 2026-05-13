"""
Script untuk memecah gambar multi-panel menjadi gambar terpisah
agar lebih mudah dibaca di skripsi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from src.config import FIGURES_PATH, FIGURE_DPI, TARGET_COLUMN, NUMERICAL_FEATURES
from src.visualization import Visualizer

# Inisialisasi
viz = Visualizer(display_inline=False)
DATA_PATH = Path("data/raw/stunting_data.csv")

def split_data_overview(df):
    """Pecah data overview menjadi 4 gambar terpisah"""
    print("\n📊 Memecah Data Overview (2x2) → 4 gambar terpisah...")
    
    # 1. Distribusi Tipe Data
    fig, ax = plt.subplots(figsize=(10, 7))
    dtype_counts = df.dtypes.astype(str).value_counts()
    colors = sns.color_palette("husl", len(dtype_counts))
    bars = ax.bar(dtype_counts.index, dtype_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Distribusi Tipe Data', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Tipe Data', fontsize=14)
    ax.set_ylabel('Jumlah Kolom', fontsize=14)
    for bar, val in zip(bars, dtype_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(val), ha='center', va='bottom', fontsize=13, fontweight='bold')
    plt.tight_layout()
    viz.save_figure(fig, '01a_distribusi_tipe_data')
    
    # 2. Missing Values
    fig, ax = plt.subplots(figsize=(10, 7))
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_cols = missing[missing > 0]
        bars = ax.barh(missing_cols.index, missing_cols.values, color='coral', edgecolor='black', linewidth=1.5)
        ax.set_title('Missing Values per Kolom', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Jumlah Missing', fontsize=14)
        for bar, val in zip(bars, missing_cols.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val} ({val/len(df)*100:.1f}%)', va='center', fontsize=12)
    else:
        ax.text(0.5, 0.5, '✅ Tidak ada Missing Values!', 
                transform=ax.transAxes, ha='center', va='center', fontsize=18, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=1))
        ax.set_title('Status Missing Values', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
    plt.tight_layout()
    viz.save_figure(fig, '01b_missing_values')
    
    # 3. Target Distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    if TARGET_COLUMN in df.columns:
        target_counts = df[TARGET_COLUMN].value_counts()
        colors = sns.color_palette("husl", len(target_counts))
        bars = ax.bar(target_counts.index, target_counts.values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_title(f'Distribusi Target: {TARGET_COLUMN}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Kelas', fontsize=14)
        ax.set_ylabel('Jumlah', fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        for bar, val in zip(bars, target_counts.values):
            pct = val / len(df) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    viz.save_figure(fig, '01c_distribusi_target')
    
    # 4. Dataset Info Summary
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
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
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes, fontsize=16,
            verticalalignment='center', horizontalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1.5))
    ax.set_title('Informasi Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    viz.save_figure(fig, '01d_ringkasan_dataset')
    
    print("✅ Data Overview dipecah menjadi 4 gambar terpisah")

def split_numerical_distributions(df):
    """Pecah numerical distributions menjadi gambar terpisah per fitur"""
    print("\n📊 Memecah Numerical Distributions → gambar terpisah per fitur...")
    
    num_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    
    for col in num_cols:
        # Histogram dan Boxplot untuk setiap fitur
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Histogram
        axes[0].hist(df[col], bins=50, color='steelblue', edgecolor='black', alpha=0.8, linewidth=1.2)
        axes[0].set_xlabel(col, fontsize=13)
        axes[0].set_ylabel('Frekuensi', fontsize=13)
        axes[0].set_title(f'Distribusi {col}', fontsize=15, fontweight='bold', pad=15)
        
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[0].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        axes[0].legend(fontsize=12)
        axes[0].tick_params(labelsize=11)
        axes[0].grid(alpha=0.3)
        
        # Boxplot
        sns.boxplot(data=df, y=col, ax=axes[1], color='lightblue', linewidth=1.5)
        axes[1].set_ylabel(col, fontsize=13)
        axes[1].set_title(f'Box Plot {col}', fontsize=15, fontweight='bold', pad=15)
        axes[1].tick_params(labelsize=11)
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        safe_name = col.replace(' ', '_').replace('(', '').replace(')', '').lower()
        viz.save_figure(fig, f'numerical_dist_{safe_name}')
    
    print(f"✅ Numerical distributions dipecah menjadi {len(num_cols)} gambar")

def split_train_test_split_viz(df, y_train, y_test, class_names):
    """Pecah train/test split visualization menjadi 3 gambar terpisah"""
    print("\n📊 Memecah Train/Test Split (1x3) → 3 gambar terpisah...")
    
    colors = sns.color_palette("husl", len(np.unique(y_train)))
    
    # 1. Overall split (Pie chart)
    fig, ax = plt.subplots(figsize=(10, 8))
    sizes = [len(y_train), len(y_test)]
    labels = [f'Training\n{len(y_train):,}\n({len(y_train)/(len(y_train)+len(y_test))*100:.1f}%)',
             f'Testing\n{len(y_test):,}\n({len(y_test)/(len(y_train)+len(y_test))*100:.1f}%)']
    ax.pie(sizes, labels=labels, colors=['#3498db', '#e74c3c'], autopct='',
           explode=[0.05, 0.05], shadow=True, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_title('Pembagian Data Training dan Testing', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    viz.save_figure(fig, '04a_train_test_split_pie')
    
    # 2. Training set distribution
    fig, ax = plt.subplots(figsize=(10, 8))
    unique, counts = np.unique(y_train, return_counts=True)
    labels_train = [class_names[i] if class_names is not None else str(i) for i in unique]
    bars = ax.bar(labels_train, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Distribusi Kelas - Training Set', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Kelas', fontsize=14)
    ax.set_ylabel('Jumlah', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{val:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    viz.save_figure(fig, '04b_training_set_distribution')
    
    # 3. Testing set distribution
    fig, ax = plt.subplots(figsize=(10, 8))
    unique, counts = np.unique(y_test, return_counts=True)
    labels_test = [class_names[i] if class_names is not None else str(i) for i in unique]
    bars = ax.bar(labels_test, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Distribusi Kelas - Testing Set', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Kelas', fontsize=14)
    ax.set_ylabel('Jumlah', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    viz.save_figure(fig, '04c_testing_set_distribution')
    
    print("✅ Train/Test Split dipecah menjadi 3 gambar terpisah")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("  MEMECAH GAMBAR MULTI-PANEL MENJADI GAMBAR TERPISAH")
    print("="*70)
    print("\nTujuan: Membuat gambar lebih besar dan mudah dibaca untuk skripsi")
    
    # Load data
    print(f"\n📂 Loading dataset dari: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        print(f"❌ Error: Dataset tidak ditemukan di {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Clean data (basic)
    df = df.drop_duplicates()
    print(f"✅ Data cleaned: {len(df):,} rows setelah hapus duplikat")
    
    # Split visualizations
    split_data_overview(df)
    split_numerical_distributions(df)
    
    # Untuk train/test split, kita perlu load hasil split
    # Karena kita tidak punya akses langsung, kita skip atau gunakan proporsi asli
    print("\n📊 Train/Test Split akan dipecah saat generate ulang...")
    print("   (Memerlukan data y_train dan y_test dari pipeline)")
    
    # Summary
    print("\n" + "="*70)
    print("  ✅ SELESAI!")
    print("="*70)
    print("\nGambar yang telah dipecah:")
    print("  1. Data Overview (2×2) → 4 gambar terpisah:")
    print("     - 01a_distribusi_tipe_data.png")
    print("     - 01b_missing_values.png")
    print("     - 01c_distribusi_target.png")
    print("     - 01d_ringkasan_dataset.png")
    print("\n  2. Numerical Distributions → 2 gambar (per fitur):")
    print("     - numerical_dist_umur_bulan.png")
    print("     - numerical_dist_tinggi_badan_cm.png")
    print("\n  3. Train/Test Split (1×3) → perlu regenerate dari pipeline")
    print("\nCatatan:")
    print("  - Gambar yang sudah ada tetap tersimpan")
    print("  - Gambar baru ditambahkan dengan suffix yang berbeda")
    print("  - Ukuran gambar diperbesar untuk keterbacaan")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
