"""
Script untuk memperbaiki gambar yang terpotong
Menggunakan padding yang lebih baik dan tidak memotong bagian penting
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

FIGURES_PATH = Path("reports/figures")
FIGURE_DPI = 300

def save_figure(fig, filename):
    """Save figure dengan DPI tinggi"""
    output_path = FIGURES_PATH / f"{filename}.png"
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', 
                facecolor='white', pad_inches=0.2)
    plt.close(fig)
    print(f"  ✅ Saved: {filename}.png")

def fix_target_distribution():
    """
    Perbaiki target_distribution.png
    Pecah dengan padding yang cukup agar tidak terpotong
    """
    print("\n📊 Memperbaiki Target Distribution (tidak potong diagram)...")
    
    img_path = FIGURES_PATH / "target_distribution.png"
    if not img_path.exists():
        print(f"  ⚠️  File tidak ditemukan: {img_path}")
        return
    
    img = Image.open(img_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    
    # Split dengan lebih hati-hati - tambahkan margin
    # Biasanya layout horizontal: bar (kiri) | pie (kanan)
    # Kita beri overlap sedikit agar tidak potong
    
    margin = int(width * 0.05)  # 5% margin
    mid_width = width // 2
    
    # Left part (bar chart) - ambil lebih banyak di kanan
    left_part = img_array[:, :mid_width + margin]
    
    # Right part (pie chart) - ambil lebih banyak di kiri
    right_part = img_array[:, mid_width - margin:]
    
    # Save dengan padding
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(left_part)
    ax.axis('off')
    plt.tight_layout(pad=0.5)
    save_figure(fig, 'target_dist_bar_fixed')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(right_part)
    ax.axis('off')
    plt.tight_layout(pad=0.5)
    save_figure(fig, 'target_dist_pie_fixed')
    
    print("  ✅ Target Distribution diperbaiki (dengan padding)")

def fix_model_summary_dashboard():
    """
    Perbaiki model_comparison_summary.png
    Pecah dengan lebih hati-hati agar metrics table tidak terpotong
    """
    print("\n📊 Memperbaiki Model Comparison Summary Dashboard...")
    
    img_path = FIGURES_PATH / "model_comparison_summary.png"
    if not img_path.exists():
        print(f"  ⚠️  File tidak ditemukan: {img_path}")
        return
    
    img = Image.open(img_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    
    # Dashboard biasanya 2 rows x 3 cols atau 3 rows x 2 cols
    # Kita coba deteksi yang mana dengan rasio
    aspect_ratio = width / height
    
    if aspect_ratio > 1.5:  # Landscape wide (2 rows x 3 cols)
        n_rows, n_cols = 2, 3
    else:  # Portrait or square (3 rows x 2 cols)
        n_rows, n_cols = 3, 2
    
    print(f"  Detected layout: {n_rows} rows x {n_cols} cols")
    
    # Bagi dengan padding antar cell
    padding_v = int(height * 0.02)  # 2% vertical padding
    padding_h = int(width * 0.02)   # 2% horizontal padding
    
    row_height = height // n_rows
    col_width = width // n_cols
    
    parts = []
    labels = ['metrics_table', 'roc_curves', 'pr_curves', 
              'confusion_matrices', 'feature_importance', 'training_time']
    
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            # Hitung boundaries dengan padding
            y_start = max(0, i * row_height - padding_v)
            y_end = min(height, (i + 1) * row_height + padding_v)
            x_start = max(0, j * col_width - padding_h)
            x_end = min(width, (j + 1) * col_width + padding_h)
            
            part = img_array[y_start:y_end, x_start:x_end]
            
            # Save dengan ukuran lebih besar
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.imshow(part)
            ax.axis('off')
            plt.tight_layout(pad=0.3)
            
            label = labels[idx] if idx < len(labels) else f'panel_{idx+1}'
            save_figure(fig, f'summary_{idx+1}_{label}_fixed')
            idx += 1
            
            if idx >= len(labels):
                break
        if idx >= len(labels):
            break
    
    print(f"  ✅ Dashboard dipecah menjadi {idx} komponen (dengan padding)")

def create_single_large_versions():
    """
    Buat versi single besar untuk target_dist dan dashboard
    Jika pecah tetap bermasalah, gunakan versi diperbesar saja
    """
    print("\n📊 Membuat versi SINGLE BESAR sebagai alternatif...")
    
    # Target Distribution - versi besar
    img_path = FIGURES_PATH / "target_distribution.png"
    if img_path.exists():
        img = Image.open(img_path)
        img_array = np.array(img)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(img_array)
        ax.axis('off')
        plt.tight_layout(pad=0.2)
        save_figure(fig, 'target_dist_LARGE')
        print("  ✅ target_dist_LARGE.png (16×8 inches)")
    
    # Model Summary - versi besar
    img_path = FIGURES_PATH / "model_comparison_summary.png"
    if img_path.exists():
        img = Image.open(img_path)
        img_array = np.array(img)
        
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.imshow(img_array)
        ax.axis('off')
        plt.tight_layout(pad=0.2)
        save_figure(fig, 'model_summary_LARGE')
        print("  ✅ model_summary_LARGE.png (18×12 inches)")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("  MEMPERBAIKI GAMBAR YANG TERPOTONG")
    print("="*70)
    print("\nMasalah:")
    print("  - target_dist_bar terpotong pada diagram batang")
    print("  - summary_metrics terpotong")
    print("\nSolusi:")
    print("  1. Pecah ulang dengan padding lebih besar")
    print("  2. Buat versi LARGE sebagai alternatif")
    
    if not FIGURES_PATH.exists():
        print(f"\n❌ Error: Folder {FIGURES_PATH} tidak ditemukan!")
        return
    
    # Fix dengan padding
    fix_target_distribution()
    fix_model_summary_dashboard()
    
    # Buat versi large sebagai backup
    create_single_large_versions()
    
    # Summary
    print("\n" + "="*70)
    print("  ✅ SELESAI!")
    print("="*70)
    print("\n📋 File yang dibuat:")
    print("\n1. Target Distribution (FIXED):")
    print("   - target_dist_bar_fixed.png (dengan padding, 12×8\")")
    print("   - target_dist_pie_fixed.png (dengan padding, 12×8\")")
    print("   - target_dist_LARGE.png (alternatif: 1 gambar besar 16×8\")")
    
    print("\n2. Model Summary Dashboard (FIXED):")
    print("   - summary_1_metrics_table_fixed.png (14×10\")")
    print("   - summary_2_roc_curves_fixed.png (14×10\")")
    print("   - summary_3_pr_curves_fixed.png (14×10\")")
    print("   - summary_4_confusion_matrices_fixed.png (14×10\")")
    print("   - summary_5_feature_importance_fixed.png (14×10\")")
    print("   - summary_6_training_time_fixed.png (14×10\")")
    print("   - model_summary_LARGE.png (alternatif: 1 gambar besar 18×12\")")
    
    print("\n💡 Rekomendasi:")
    print("   - Cek versi *_fixed.png terlebih dahulu")
    print("   - Jika masih terpotong, gunakan versi *_LARGE.png")
    print("   - Versi LARGE ukuran lebih besar tapi tidak dipecah")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
