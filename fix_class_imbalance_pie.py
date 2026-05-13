"""
Script untuk memperbaiki Class Imbalance Pie Chart yang terpotong
Buat versi LARGE dengan padding ekstra
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

FIGURES_PATH = Path("reports/figures")
FIGURE_DPI = 300

def save_figure(fig, filename):
    """Save figure dengan DPI tinggi dan padding ekstra"""
    output_path = FIGURES_PATH / f"{filename}.png"
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', 
                facecolor='white', pad_inches=0.5)  # Padding ekstra!
    plt.close(fig)
    print(f"  ✅ Saved: {filename}.png")

def fix_class_imbalance_pie():
    """
    Memperbaiki 03b_class_imbalance_pie.png yang terpotong
    """
    print("\n📊 Memperbaiki Class Imbalance Pie Chart...")
    
    # Gunakan gambar asli yang lengkap
    img_path_original = FIGURES_PATH / "03_class_imbalance.png"
    
    if not img_path_original.exists():
        print(f"  ⚠️  File asli tidak ditemukan: {img_path_original}")
        return
    
    img = Image.open(img_path_original)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    
    # Split dengan margin lebih besar (10% instead of 5%)
    margin = int(width * 0.1)  # 10% margin
    mid_width = width // 2
    
    # Right part (pie chart) - ambil dari tengah ke kanan dengan margin besar
    right_part = img_array[:, mid_width - margin:]
    
    # Save dengan figsize lebih besar dan padding ekstra
    fig, ax = plt.subplots(figsize=(14, 10))  # Lebih besar dari 12x8
    ax.imshow(right_part)
    ax.axis('off')
    plt.tight_layout(pad=1.0)  # Padding lebih besar
    save_figure(fig, '03b_class_imbalance_pie_FIXED')
    
    print("  ✅ Class Imbalance Pie Chart diperbaiki (14×10 inches dengan padding)")
    
    # Buat juga versi LARGE dari gambar lengkap
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(img_array)
    ax.axis('off')
    plt.tight_layout(pad=0.5)
    save_figure(fig, '03_class_imbalance_LARGE')
    
    print("  ✅ Alternatif: Class Imbalance LARGE (bar + pie, 16×10 inches)")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("  MEMPERBAIKI CLASS IMBALANCE PIE CHART")
    print("="*70)
    print("\nMasalah: Gambar 3.5 (Pie Chart) terpotong")
    print("Solusi: Buat ulang dengan padding ekstra dan ukuran lebih besar")
    
    if not FIGURES_PATH.exists():
        print(f"\n❌ Error: Folder {FIGURES_PATH} tidak ditemukan!")
        return
    
    fix_class_imbalance_pie()
    
    # Summary
    print("\n" + "="*70)
    print("  ✅ SELESAI!")
    print("="*70)
    print("\n📋 File yang dibuat:")
    print("\n1. 03b_class_imbalance_pie_FIXED.png")
    print("   - Ukuran: 14×10 inches")
    print("   - Padding: Ekstra besar (tidak terpotong)")
    print("   - Rekomendasi: GUNAKAN INI untuk skripsi ⭐")
    
    print("\n2. 03_class_imbalance_LARGE.png")
    print("   - Ukuran: 16×10 inches")
    print("   - Isi: Bar + Pie dalam 1 gambar")
    print("   - Alternatif: Jika pie saja masih terpotong")
    
    print("\n💡 Update dokumen Word:")
    print("   Ganti 03b_class_imbalance_pie.png")
    print("   dengan 03b_class_imbalance_pie_FIXED.png")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
