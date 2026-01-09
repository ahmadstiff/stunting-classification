# Klasifikasi Biner Status Stunting dengan Ensemble Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Deskripsi

Proyek ini merupakan implementasi skripsi dengan judul:

> **"Analisis Komparasi Kinerja Algoritma Ensemble Learning (Random Forest, XGBoost, LightGBM, dan CatBoost) untuk Klasifikasi Biner Status Stunting"**

Sistem ini melakukan klasifikasi biner untuk mendeteksi status stunting pada balita menggunakan 4 algoritma ensemble learning, kemudian membandingkan performa masing-masing algoritma.

### Klasifikasi Biner
- **0 = Tidak Stunting** (normal + tinggi)
- **1 = Stunting** (stunted + severely stunted)

## 🎯 Algoritma yang Digunakan

| No | Algoritma | Library |
|----|-----------|---------|
| 1 | Random Forest | scikit-learn |
| 2 | XGBoost | xgboost |
| 3 | LightGBM | lightgbm |
| 4 | CatBoost | catboost |

## 📊 Dataset

Dataset yang digunakan adalah data status gizi balita dari Kaggle:
- **Sumber**: [Rendy Rian - Stunting Dataset](https://www.kaggle.com/datasets/rendyrian/stunting-balita-detection-121k-rows)
- **Jumlah Data**: 120,999 rows (sebelum cleaning)
- **Jumlah Data Bersih**: 39,425 rows (setelah hapus duplikat)

### Fitur yang Digunakan
| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| Jenis Kelamin | Kategorikal | Jenis kelamin balita (laki-laki/perempuan) |
| Umur (bulan) | Numerik | Usia balita dalam bulan |
| Tinggi Badan (cm) | Numerik | Tinggi badan balita dalam cm |

### Target Variable
| Nilai Asli | Mapping | Label |
|------------|---------|-------|
| normal | 0 | Tidak Stunting |
| tinggi | 0 | Tidak Stunting |
| stunted | 1 | Stunting |
| severely stunted | 1 | Stunting |

## 📁 Struktur Proyek

```
stunting/
├── data/
│   ├── raw/                    # Data mentah dari Kaggle
│   │   └── stunting_data.csv
│   └── processed/              # Data yang sudah diproses
├── models/                     # Model yang sudah ditraining (pickle)
├── reports/
│   ├── figures/                # Visualisasi hasil
│   │   ├── model_comparison_summary.png
│   │   ├── confusion_matrix_comparison.png
│   │   ├── roc_curves_binary.png
│   │   ├── precision_recall_curves.png
│   │   └── ...
│   └── model_comparison.csv    # Hasil perbandingan model
├── src/
│   ├── __init__.py
│   ├── config.py               # Konfigurasi dan hyperparameters
│   ├── preprocessing.py        # Preprocessing dan encoding
│   ├── models.py               # Training dan evaluasi model
│   └── visualization.py        # Visualisasi dan plotting
├── main.py                     # Script utama pipeline
├── requirements.txt            # Dependencies
├── view_images.py              # Viewer untuk gambar hasil
└── README.md
```

## 🚀 Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd stunting
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Kaggle API (Opsional)
Jika ingin download dataset otomatis:
```bash
pip install kagglehub
# Pastikan ~/.kaggle/kaggle.json sudah dikonfigurasi
```

Atau download manual dari Kaggle dan letakkan di `data/raw/stunting_data.csv`

## 💻 Cara Penggunaan

### Menjalankan Pipeline Lengkap

```bash
# Dengan hyperparameter tuning (lebih lama, hasil lebih optimal)
python main.py

# Tanpa hyperparameter tuning (lebih cepat)
python main.py --no-tuning

# Tanpa inline image display
python main.py --no-inline

# Kombinasi
python main.py --no-tuning --no-inline
```

### Opsi Command Line

| Opsi | Deskripsi |
|------|-----------|
| `--no-tuning` | Skip hyperparameter tuning (lebih cepat) |
| `--no-inline` | Disable inline image display di terminal |

### Output Pipeline

Pipeline akan menghasilkan:
1. **Visualisasi** di `reports/figures/`
2. **Hasil perbandingan** di `reports/model_comparison.csv`
3. **Terminal output** dengan metrics dan ranking

## 📈 Hasil Eksperimen

### Perbandingan Performa Model

| Ranking | Model | Accuracy | F1-Score | ROC-AUC |
|---------|-------|----------|----------|---------|
| 🥇 | **Random Forest** | **99.75%** | **99.75%** | **99.997%** |
| 🥈 | LightGBM | 99.19% | 99.19% | 99.97% |
| 🥉 | CatBoost | 98.88% | 98.88% | 99.93% |
| 4️⃣ | XGBoost | 98.50% | 98.50% | 99.91% |

### Feature Importance

Berdasarkan hasil analisis, fitur yang paling berpengaruh:
1. **Tinggi Badan (cm)** - ~63.55%
2. **Umur (bulan)** - ~36.27%
3. **Jenis Kelamin** - ~0.18%

### Visualisasi Hasil

| Visualisasi | Deskripsi |
|-------------|-----------|
| `model_comparison_summary.png` | Dashboard perbandingan lengkap 4 model |
| `confusion_matrix_comparison.png` | Perbandingan confusion matrix |
| `roc_curves_binary.png` | ROC curves untuk binary classification |
| `precision_recall_curves.png` | Precision-Recall curves |
| `feature_importance_comparison_multi.png` | Perbandingan feature importance |
| `cv_results_comparison.png` | Hasil cross-validation |

## 🔧 Konfigurasi

Konfigurasi dapat diubah di `src/config.py`:

### Hyperparameters Default

```python
# Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'criterion': 'entropy',  # C4.5-like
    'max_depth': None,
    'random_state': 42
}

# XGBoost
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42
}

# LightGBM
LGBM_PARAMS = {
    'n_estimators': 100,
    'max_depth': -1,
    'learning_rate': 0.1,
    'random_state': 42
}

# CatBoost
CB_PARAMS = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbose': False
}
```

### Parameter Grid untuk Tuning

Parameter grid untuk hyperparameter tuning juga dapat dikonfigurasi di `src/config.py`.

## 📊 Metrik Evaluasi

Metrik yang digunakan untuk evaluasi:

| Metrik | Deskripsi |
|--------|-----------|
| **Accuracy** | Proporsi prediksi yang benar |
| **Precision** | Ketepatan prediksi positif |
| **Recall** | Kemampuan mendeteksi kelas positif |
| **F1-Score** | Harmonic mean dari Precision dan Recall |
| **ROC-AUC** | Area under ROC curve |

## 🛠️ Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
kagglehub>=0.1.0
imgcat>=0.5.0
```

## 📝 Metodologi

### Pipeline Eksperimen

1. **Data Loading** - Load dataset dari Kaggle
2. **Data Exploration** - Analisis statistik deskriptif
3. **Data Cleaning** - Hapus duplikat dan handle missing values
4. **Feature Encoding** - Encode kategorikal ke numerik
5. **Binary Target Mapping** - Konversi 4 kelas ke 2 kelas
6. **Train/Test Split** - 80% training, 20% testing
7. **Model Training** - Training 4 algoritma ensemble
8. **Cross-Validation** - 10-fold stratified CV
9. **Hyperparameter Tuning** - Grid search (opsional)
10. **Model Evaluation** - Hitung metrics dan visualisasi
11. **Model Comparison** - Ranking dan analisis komparatif

### Strategi Validasi

- **Stratified K-Fold Cross-Validation** (k=10)
- **Train/Test Split**: 80/20 dengan stratifikasi
- **Random State**: 42 (untuk reproduksibilitas)

## 👤 Author

**[Nama Anda]**
- Skripsi S1 Informatika/Ilmu Komputer
- Tahun: 2024/2025

## 📄 License

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

## 🙏 Acknowledgements

- Dataset: [Rendy Rian - Kaggle](https://www.kaggle.com/datasets/rendyrian/stunting-balita-detection-121k-rows)
- Dosen Pembimbing: [Nama Dosen]
- Libraries: scikit-learn, XGBoost, LightGBM, CatBoost

---

<p align="center">
  <i>Dibuat dengan ❤️ untuk Skripsi</i>
</p>
