"""
Script untuk memecah gambar multi-panel menjadi gambar terpisah
Fokus pada: 3.6, 4.1, 4.7, 4.8, 4.9
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
from PIL import Image

FIGURES_PATH = Path("reports/figures")
FIGURE_DPI = 300

def save_figure(fig, filename):
    """Save figure dengan DPI tinggi"""
    output_path = FIGURES_PATH / f"{filename}.png"
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Saved: {filename}.png")

def split_class_imbalance():
    """
    Memecah 03_class_imbalance.png (Gambar 3.6)
    Biasanya berisi: bar chart + pie chart
    """
    print("\n📊 Memecah Gambar 3.6: Class Imbalance (bar + pie)...")
    
    img_path = FIGURES_PATH / "03_class_imbalance.png"
    if not img_path.exists():
        print(f"  ⚠️  File tidak ditemukan: {img_path}")
        return
    
    img = Image.open(img_path)
    img_array = np.array(img)
    
    # Hitung dimensi untuk split horizontal (bar kiri, pie kanan)
    height, width = img_array.shape[:2]
    mid_width = width // 2
    
    # Split menjadi 2 bagian
    left_part = img_array[:, :mid_width]
    right_part = img_array[:, mid_width:]
    
    # Save bar chart (kiri)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(left_part)
    ax.axis('off')
    plt.tight_layout(pad=0)
    save_figure(fig, '03a_class_imbalance_bar')
    
    # Save pie chart (kanan)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(right_part)
    ax.axis('off')
    plt.tight_layout(pad=0)
    save_figure(fig, '03b_class_imbalance_pie')
    
    print("  ✅ Class Imbalance dipecah: bar (3a) + pie (3b)")

def split_target_distribution():
    """
    Memecah target_distribution.png (Gambar 4.1)
    Biasanya berisi: bar chart + pie chart
    """
    print("\n📊 Memecah Gambar 4.1: Target Distribution (bar + pie)...")
    
    img_path = FIGURES_PATH / "target_distribution.png"
    if not img_path.exists():
        print(f"  ⚠️  File tidak ditemukan: {img_path}")
        return
    
    img = Image.open(img_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    mid_width = width // 2
    
    left_part = img_array[:, :mid_width]
    right_part = img_array[:, mid_width:]
    
    # Save bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(left_part)
    ax.axis('off')
    plt.tight_layout(pad=0)
    save_figure(fig, 'target_dist_bar')
    
    # Save pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(right_part)
    ax.axis('off')
    plt.tight_layout(pad=0)
    save_figure(fig, 'target_dist_pie')
    
    print("  ✅ Target Distribution dipecah: bar + pie")

def split_cv_results():
    """
    Memecah cv_results_comparison.png (Gambar 4.7)
    Biasanya berisi: multiple subplots (metrics comparison)
    """
    print("\n📊 Memecah Gambar 4.7: CV Results Comparison...")
    
    img_path = FIGURES_PATH / "cv_results_comparison.png"
    if not img_path.exists():
        print(f"  ⚠️  File tidak ditemukan: {img_path}")
        return
    
    img = Image.open(img_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    
    # Jika layout 2x2 atau 2x3
    # Kita pecah berdasarkan grid
    # Untuk sekarang kita buat 1 versi besar dulu
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img_array)
    ax.axis('off')
    plt.tight_layout(pad=0)
    save_figure(fig, 'cv_results_large')
    
    print("  ✅ CV Results diperbesar menjadi 1 gambar besar (14×10 inches)")

def split_model_comparison_summary():
    """
    Memecah model_comparison_summary.png (Gambar 4.8)
    Dashboard dengan banyak panel - pecah menjadi komponen-komponen
    """
    print("\n📊 Memecah Gambar 4.8: Model Comparison Summary (Dashboard)...")
    
    img_path = FIGURES_PATH / "model_comparison_summary.png"
    if not img_path.exists():
        print(f"  ⚠️  File tidak ditemukan: {img_path}")
        return
    
    img = Image.open(img_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    
    # Dashboard biasanya punya layout 2x3 atau 3x2
    # Kita pecah ke 6 bagian
    
    # Coba grid 2 rows x 3 cols
    row_height = height // 2
    col_width = width // 3
    
    parts = []
    labels = ['metrics_table', 'roc_curves', 'pr_curves', 
              'confusion_matrices', 'feature_importance', 'training_time']
    
    for i in range(2):  # rows
        for j in range(3):  # cols
            y_start = i * row_height
            y_end = (i + 1) * row_height if i < 1 else height
            x_start = j * col_width
            x_end = (j + 1) * col_width if j < 2 else width
            
            part = img_array[y_start:y_end, x_start:x_end]
            parts.append(part)
    
    # Save each part
    for idx, (part, label) in enumerate(zip(parts, labels)):
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(part)
        ax.axis('off')
        plt.tight_layout(pad=0)
        save_figure(fig, f'summary_{idx+1}_{label}')
    
    print(f"  ✅ Dashboard dipecah menjadi {len(parts)} komponen terpisah")

def split_confusion_matrix_comparison():
    """
    Memecah confusion_matrix_comparison.png (Gambar 4.9)
    Grid 3×2 (6 model) → 6 gambar terpisah
    """
    print("\n📊 Memecah Gambar 4.9: Confusion Matrix Comparison (3×2 grid)...")
    
    img_path = FIGURES_PATH / "confusion_matrix_comparison.png"
    if not img_path.exists():
        print(f"  ⚠️  File tidak ditemukan: {img_path}")
        return
    
    img = Image.open(img_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    
    # Grid 3 rows x 2 cols (6 models)
    n_rows = 3
    n_cols = 2
    
    row_height = height // n_rows
    col_width = width // n_cols
    
    models = ['Decision Tree', 'Random Forest', 'XGBoost', 
              'LightGBM', 'CatBoost', 'Extra Model']
    
    parts = []
    for i in range(n_rows):
        for j in range(n_cols):
            y_start = i * row_height
            y_end = (i + 1) * row_height if i < n_rows - 1 else height
            x_start = j * col_width
            x_end = (j + 1) * col_width if j < n_cols - 1 else width
            
            part = img_array[y_start:y_end, x_start:x_end]
            parts.append(part)
    
    # Save each confusion matrix
    model_names = ['dt', 'rf', 'xgb', 'lgbm', 'catboost', 'model6']
    
    for idx, (part, model_name) in enumerate(zip(parts[:5], model_names[:5])):
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(part)
        ax.axis('off')
        plt.tight_layout(pad=0)
        save_figure(fig, f'cm_{idx+1}_{model_name}')
    
    print(f"  ✅ Confusion Matrices dipecah menjadi 5 gambar terpisah per model")

def split_feature_importance_comparison():
    """
    Memecah feature_importance_comparison_multi.png
    Grid 3×2 → 6 gambar terpisah
    """
    print("\n📊 Memecah Feature Importance Comparison (3×2 grid)...")
    
    img_path = FIGURES_PATH / "feature_importance_comparison_multi.png"
    if not img_path.exists():
        print(f"  ⚠️  File tidak ditemukan: {img_path}")
        return
    
    img = Image.open(img_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    
    n_rows = 3
    n_cols = 2
    
    row_height = height // n_rows
    col_width = width // n_cols
    
    parts = []
    for i in range(n_rows):
        for j in range(n_cols):
            y_start = i * row_height
            y_end = (i + 1) * row_height if i < n_rows - 1 else height
            x_start = j * col_width
            x_end = (j + 1) * col_width if j < n_cols - 1 else width
            
            part = img_array[y_start:y_end, x_start:x_end]
            parts.append(part)
    
    model_names = ['dt', 'rf', 'xgb', 'lgbm', 'catboost', 'model6']
    
    for idx, (part, model_name) in enumerate(zip(parts[:5], model_names[:5])):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(part)
        ax.axis('off')
        plt.tight_layout(pad=0)
        save_figure(fig, f'fi_{idx+1}_{model_name}')
    
    print(f"  ✅ Feature Importance dipecah menjadi 5 gambar terpisah per model")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("  MEMECAH LEBIH BANYAK GAMBAR UNTUK SKRIPSI")
    print("="*70)
    print("\nTarget gambar:")
    print("  - Gambar 3.6: Class Imbalance (bar + pie)")
    print("  - Gambar 4.1: Target Distribution (bar + pie)")
    print("  - Gambar 4.7: CV Results")
    print("  - Gambar 4.8: Model Comparison Summary (Dashboard)")
    print("  - Gambar 4.9: Confusion Matrix Comparison (grid 3×2)")
    print("  - Bonus: Feature Importance Comparison (grid 3×2)")
    
    if not FIGURES_PATH.exists():
        print(f"\n❌ Error: Folder {FIGURES_PATH} tidak ditemukan!")
        return
    
    # Split images
    split_class_imbalance()
    split_target_distribution()
    split_cv_results()
    split_model_comparison_summary()
    split_confusion_matrix_comparison()
    split_feature_importance_comparison()
    
    # Summary
    print("\n" + "="*70)
    print("  ✅ SELESAI!")
    print("="*70)
    print("\n📋 Ringkasan gambar yang dipecah:")
    print("\n1. Gambar 3.6 (Class Imbalance):")
    print("   - 03a_class_imbalance_bar.png")
    print("   - 03b_class_imbalance_pie.png")
    
    print("\n2. Gambar 4.1 (Target Distribution):")
    print("   - target_dist_bar.png")
    print("   - target_dist_pie.png")
    
    print("\n3. Gambar 4.7 (CV Results):")
    print("   - cv_results_large.png (diperbesar)")
    
    print("\n4. Gambar 4.8 (Dashboard):")
    print("   - summary_1_metrics_table.png")
    print("   - summary_2_roc_curves.png")
    print("   - summary_3_pr_curves.png")
    print("   - summary_4_confusion_matrices.png")
    print("   - summary_5_feature_importance.png")
    print("   - summary_6_training_time.png")
    
    print("\n5. Gambar 4.9 (Confusion Matrices):")
    print("   - cm_1_dt.png (Decision Tree)")
    print("   - cm_2_rf.png (Random Forest)")
    print("   - cm_3_xgb.png (XGBoost)")
    print("   - cm_4_lgbm.png (LightGBM)")
    print("   - cm_5_catboost.png (CatBoost)")
    
    print("\n6. Bonus (Feature Importance):")
    print("   - fi_1_dt.png sampai fi_5_catboost.png")
    
    print("\n💡 Semua gambar diperbesar untuk keterbacaan optimal!")
    print("   Ukuran: 10-14 inches, DPI 300")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
