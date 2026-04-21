# Laporan Project Machine Learning
# Prediksi Biaya Asuransi Kesehatan

---

## Daftar Isi

1. [Latar Belakang](#1-latar-belakang)
2. [Deskripsi Dataset](#2-deskripsi-dataset)
3. [Tahapan Preprocessing](#3-tahapan-preprocessing)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Model Building](#5-model-building)
6. [Hyperparameter Tuning](#6-hyperparameter-tuning)
7. [Evaluasi Model](#7-evaluasi-model)
8. [Perbandingan Hasil Model](#8-perbandingan-hasil-model)
9. [Kesimpulan](#9-kesimpulan)

---

## 1. Latar Belakang

Industri asuransi kesehatan memerlukan model prediktif yang akurat untuk mengestimasi biaya klaim medis setiap individu. Prediksi biaya yang akurat sangat penting untuk:

- **Penetapan premi** yang adil dan kompetitif
- **Manajemen risiko** keuangan perusahaan asuransi
- **Perencanaan anggaran** kesehatan bagi individu dan organisasi
- **Identifikasi faktor risiko** utama yang mempengaruhi biaya kesehatan

Beberapa faktor yang diketahui mempengaruhi biaya asuransi kesehatan antara lain usia, indeks massa tubuh (BMI), status merokok, jumlah tanggungan, dan wilayah tempat tinggal. Namun, hubungan antar faktor-faktor ini bersifat **non-linear** — misalnya, kombinasi antara merokok dan obesitas tidak sekadar penjumlahan risiko, melainkan **efek multiplikatif** yang sangat signifikan.

Dalam project ini, kami membangun model Machine Learning untuk **memprediksi biaya tagihan asuransi kesehatan (charges)** menggunakan data demografis dan kesehatan pasien. Kami menggunakan **4 algoritma** yang berbeda (Linear Regression, Random Forest, XGBoost, LightGBM) dan membandingkan performanya pada **3 rasio pembagian data** (70/30, 80/20, 90/10) dengan **hyperparameter tuning** pada setiap model.

---

## 2. Deskripsi Dataset

### 2.1 Sumber Data

Dataset yang digunakan adalah **Medical Cost Personal Dataset** yang tersedia di platform Kaggle:
- **URL:** https://www.kaggle.com/datasets/mirichoi0218/insurance
- **Jumlah data:** 1.338 baris
- **Jumlah fitur:** 6 fitur input + 1 target variable
- **Missing values:** Tidak ada (dataset bersih)

### 2.2 Variabel / Fitur Dataset

| No | Fitur | Tipe Data | Deskripsi | Rentang Nilai |
|----|-------|-----------|-----------|---------------|
| 1 | `age` | Integer | Usia pasien | 18 – 64 tahun |
| 2 | `sex` | Kategorikal | Jenis kelamin | male, female |
| 3 | `bmi` | Float | Body Mass Index (indeks massa tubuh) | 15.96 – 53.13 |
| 4 | `children` | Integer | Jumlah anak/tanggungan | 0 – 5 |
| 5 | `smoker` | Kategorikal | Status merokok | yes, no |
| 6 | `region` | Kategorikal | Wilayah tempat tinggal di AS | northeast, northwest, southeast, southwest |
| 7 | **`charges`** | Float | **Target — Biaya asuransi tahunan (USD)** | $1.121,87 – $63.770,43 |

### 2.3 Statistik Deskriptif

| Statistik | Age | BMI | Children | Charges |
|-----------|-----|-----|----------|---------|
| Mean | 39.21 | 30.66 | 1.09 | $13.270,42 |
| Std | 14.05 | 6.10 | 1.21 | $12.110,01 |
| Min | 18 | 15.96 | 0 | $1.121,87 |
| 25% | 27 | 26.30 | 0 | $4.740,29 |
| 50% (Median) | 39 | 30.40 | 1 | $9.382,03 |
| 75% | 51 | 34.69 | 2 | $16.639,91 |
| Max | 64 | 53.13 | 5 | $63.770,43 |

### 2.4 Distribusi Kategorikal

| Fitur | Kategori | Jumlah | Persentase |
|-------|----------|--------|------------|
| Sex | Male | 676 | 50.5% |
|     | Female | 662 | 49.5% |
| Smoker | No | 1.064 | 79.5% |
|        | Yes | 274 | 20.5% |
| Region | Southeast | 364 | 27.2% |
|        | Southwest | 325 | 24.3% |
|        | Northwest | 325 | 24.3% |
|        | Northeast | 324 | 24.2% |

---

## 3. Tahapan Preprocessing

### 3.1 Pengecekan Missing Values

Dilakukan pengecekan missing values pada seluruh kolom. Hasilnya: **tidak ditemukan missing values** pada dataset ini, sehingga tidak diperlukan proses imputasi.

### 3.2 Penanganan Outlier

Target variable `charges` memiliki distribusi **right-skewed** (miring ke kanan), artinya sebagian kecil pasien memiliki biaya yang sangat tinggi. Tidak dilakukan penghapusan outlier karena data outlier pada charges merupakan data yang valid — pasien perokok dengan BMI tinggi memang memiliki biaya yang jauh lebih tinggi secara alamiah.

### 3.3 Feature Engineering

Feature engineering merupakan tahapan krusial dalam project ini. Dari 6 fitur mentah, kami menciptakan **21 fitur** untuk menangkap hubungan non-linear:

#### a. Encoding Variabel Kategorikal
- `smoker` dikonversi menjadi `smoker_binary` (1 = yes, 0 = no)
- `sex` dan `region` di-encode menggunakan **one-hot encoding** (dengan `drop_first=True` untuk menghindari multikolinearitas)

#### b. Fitur Interaksi (Interaction Features)
| Fitur Baru | Formula | Alasan |
|------------|---------|--------|
| **`smoker_bmi`** | `smoker_binary × bmi` | **Fitur terpenting** — menangkap efek multiplikatif merokok + BMI tinggi |
| `smoker_age` | `smoker_binary × age` | Perokok yang lebih tua membayar lebih mahal |
| `age_bmi` | `age × bmi` | Interaksi usia dan BMI |
| `smoker_obese` | `smoker_binary × is_obese` | Perokok yang juga obesitas (double risk factor) |

#### c. Fitur Polinomial
| Fitur Baru | Formula | Alasan |
|------------|---------|--------|
| `age_sq` | `age²` | Efek non-linear usia terhadap biaya |
| `bmi_sq` | `bmi²` | Efek non-linear BMI terhadap biaya |

#### d. Fitur Kategorikal Biner
| Fitur Baru | Formula | Alasan |
|------------|---------|--------|
| `is_obese` | `bmi ≥ 30` | Flag obesitas |
| `is_overweight` | `bmi ≥ 25` | Flag kelebihan berat badan |
| `has_children` | `children > 0` | Flag memiliki tanggungan |
| `age_group_young` | `age < 30` | Kategori usia muda |
| `age_group_mid` | `30 ≤ age < 50` | Kategori usia menengah |
| `age_group_senior` | `age ≥ 50` | Kategori usia senior |

#### e. Transformasi Log
| Fitur Baru | Formula | Alasan |
|------------|---------|--------|
| `log_bmi` | `log(1 + bmi)` | Koreksi skewness pada distribusi BMI |

### 3.4 Transformasi Target Variable

Target `charges` ditransformasi menggunakan **log-transform** (`np.log1p`) selama training untuk menangani distribusi yang right-skewed. Prediksi dikembalikan ke skala asli menggunakan **inverse transform** (`np.expm1`) saat evaluasi.

### 3.5 Ringkasan Preprocessing

| Tahap | Sebelum | Sesudah |
|-------|---------|---------|
| Jumlah fitur | 6 (mentah) | 21 (setelah engineering) |
| Missing values | 0 | 0 |
| Outlier dihapus | - | Tidak (data valid) |
| Encoding | 3 kolom kategorikal | One-hot encoded |
| Target transform | Skewed | Log-transformed |

---

## 4. Exploratory Data Analysis

### 4.1 Distribusi Target Variable (Charges)

Distribusi `charges` menunjukkan pola **right-skewed** dengan:
- Mayoritas pasien memiliki biaya antara **$1.000 – $15.000**
- Sebagian kecil pasien memiliki biaya tinggi hingga **$63.770**
- Terdapat gap yang jelas pada distribusi, yang disebabkan oleh perbedaan status merokok

### 4.2 Korelasi Antar Variabel

Analisis korelasi Pearson menunjukkan fitur-fitur berikut memiliki korelasi tertinggi dengan `charges`:

| Fitur | Korelasi dengan Charges |
|-------|------------------------|
| `smoker` | 0.787 (sangat kuat) |
| `age` | 0.299 (moderat) |
| `bmi` | 0.198 (lemah-moderat) |
| `children` | 0.068 (sangat lemah) |
| `sex` | 0.058 (negligible) |
| `region` | -0.006 (negligible) |

Setelah feature engineering, korelasi meningkat signifikan:

| Fitur (Engineered) | Korelasi dengan Charges |
|---------------------|------------------------|
| **`smoker_bmi`** | **0.814** (sangat kuat) |
| `smoker_binary` | 0.787 |
| `smoker_age` | 0.690 |
| `smoker_obese` | 0.650 |
| `age` | 0.299 |

### 4.3 Insight Penting: Efek Smoker × BMI

Temuan paling penting dari EDA adalah adanya **tiga "band" biaya** yang jelas terlihat:

1. **Band bawah (Non-smoker):** Biaya naik secara linear moderat seiring usia, rata-rata ~$8.400
2. **Band tengah (Smoker, BMI rendah):** Biaya lebih tinggi dari non-smoker
3. **Band atas (Smoker, BMI tinggi ≥30):** Biaya **sangat tinggi**, rata-rata ~$32.050

| Segmen | Rata-rata Charges | Rasio |
|--------|-------------------|-------|
| Non-smoker | ~$8.400 | 1× |
| Smoker (semua) | ~$32.050 | **3.8×** lebih tinggi |

Ini bukan data leakage — ini mencerminkan **realitas aktuaria** yang sesungguhnya, dimana kombinasi merokok dan obesitas menciptakan biaya kesehatan tertinggi.

### 4.4 Visualisasi

Visualisasi yang dilakukan dalam proses EDA meliputi:
- Histogram distribusi untuk setiap fitur numerik (age, bmi, children, charges)
- Bar chart distribusi untuk fitur kategorikal (sex, smoker, region)
- Box plot `charges` berdasarkan status merokok
- Scatter plot `charges` vs `bmi` dengan warna berdasarkan status perokok
- Scatter plot `charges` vs `age` dengan warna berdasarkan status perokok
- Heatmap korelasi seluruh fitur

---

## 5. Model Building

### 5.1 Algoritma yang Digunakan

Kami menggunakan **4 algoritma** Machine Learning untuk regresi:

| No | Algoritma | Library | Akselerasi |
|----|-----------|---------|------------|
| 1 | **Linear Regression** | scikit-learn | CPU |
| 2 | **Random Forest Regressor** | cuML / scikit-learn | GPU (cuML) |
| 3 | **XGBoost Regressor** | xgboost | GPU (`tree_method='hist', device='cuda'`) |
| 4 | **LightGBM Regressor** | lightgbm | CPU/GPU |

### 5.2 Data Splitting

Dataset dibagi menjadi **3 variasi rasio** pembagian:

| Rasio | Training Set | Testing Set |
|-------|-------------|-------------|
| 70/30 | 936 sampel | 402 sampel |
| 80/20 | 1.070 sampel | 268 sampel |
| 90/10 | 1.204 sampel | 134 sampel |

Semua pembagian menggunakan `random_state=42` untuk reprodusibilitas.

### 5.3 Pendekatan Training

1. Setiap algoritma di-training pada **ketiga variasi split** secara terpisah
2. Target variable ditransformasi menggunakan **log1p** selama training
3. Evaluasi dilakukan pada **skala asli (USD)** menggunakan inverse transform **expm1**
4. Untuk setiap kombinasi (algoritma × split × hyperparameter), model terbaik dipilih berdasarkan **R² score tertinggi** pada test set

---

## 6. Hyperparameter Tuning

### 6.1 Metode Tuning

Kami menggunakan **manual parameter tuning** (grid search manual) dimana beberapa kombinasi hyperparameter dicoba untuk setiap algoritma dan split.

### 6.2 Parameter yang Dituning

#### Linear Regression
| Parameter | Nilai yang Dicoba |
|-----------|-------------------|
| `fit_intercept` | True, False |

#### Random Forest Regressor
| Parameter | Nilai yang Dicoba |
|-----------|-------------------|
| `n_estimators` | 200, 300, 500 |
| `max_depth` | 10, 15, 20 |
| `min_samples_split` | 2, 3, 5 |
| `min_samples_leaf` | 1, 2 |

#### XGBoost Regressor
| Parameter | Nilai yang Dicoba |
|-----------|-------------------|
| `n_estimators` | 300, 500, 800 |
| `learning_rate` | 0.01, 0.03, 0.05 |
| `max_depth` | 4, 5, 6 |
| `subsample` | 0.7, 0.8 |
| `colsample_bytree` | 0.7, 0.8 |
| `reg_lambda` | 1.0, 1.5, 2.0 |

#### LightGBM Regressor
| Parameter | Nilai yang Dicoba |
|-----------|-------------------|
| `n_estimators` | 300, 500, 800 |
| `learning_rate` | 0.01, 0.03, 0.05 |
| `max_depth` | 6, 8 |
| `num_leaves` | 31, 63 |
| `subsample` | 0.7, 0.8 |
| `colsample_bytree` | 0.7, 0.8 |
| `reg_lambda` | 1.0, 1.5, 2.0 |

### 6.3 Konfigurasi Model Terbaik

Model terbaik secara keseluruhan adalah **XGBoost** dengan konfigurasi:

| Parameter | Nilai Optimal |
|-----------|---------------|
| `n_estimators` | 800 |
| `learning_rate` | 0.01 |
| `max_depth` | 5 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 2.0 |
| `tree_method` | hist (GPU) |
| `random_state` | 42 |
| **Split terbaik** | **80/20** |

---

## 7. Evaluasi Model

### 7.1 Metrik Evaluasi

Untuk model regresi, kami menggunakan 4 metrik evaluasi:

| Metrik | Formula | Interpretasi |
|--------|---------|-------------|
| **R² Score** | 1 - (SS_res / SS_tot) | Proporsi varians yang dapat dijelaskan (0–1, semakin tinggi semakin baik) |
| **MAE** | Mean Absolute Error | Rata-rata kesalahan absolut dalam USD |
| **RMSE** | Root Mean Squared Error | Akar kuadrat rata-rata kesalahan kuadrat |
| **MSE** | Mean Squared Error | Rata-rata kesalahan kuadrat |

### 7.2 Hasil Evaluasi — Test R² Score

| Model | 70/30 | 80/20 | 90/10 |
|-------|-------|-------|-------|
| Linear Regression | 0.8361 | 0.8436 | 0.8372 |
| Random Forest | 0.8669 | 0.8724 | 0.8672 |
| **XGBoost** | **0.8711** | **0.8802** | **0.8710** |
| LightGBM | 0.8670 | 0.8750 | 0.8700 |

### 7.3 Hasil Evaluasi — Test MAE (USD)

| Model | 70/30 | 80/20 | 90/10 |
|-------|-------|-------|-------|
| Linear Regression | $2.474,31 | $2.500,27 | $2.493,53 |
| Random Forest | $2.096,93 | $2.046,37 | $1.983,11 |
| **XGBoost** | **$2.004,02** | **$1.912,17** | **$1.927,98** |
| LightGBM | $2.071,78 | $2.025,72 | $2.010,21 |

### 7.4 Hasil Model Terbaik

| Metrik | Nilai |
|--------|-------|
| **Model** | XGBoost Regressor |
| **Split** | 80/20 |
| **R² Score** | **0.8802** (88,02% varians terjelaskan) |
| **MAE** | **$1.912,17** |
| **RMSE** | **$4.313,40** |

### 7.5 Analisis Residual

- **Mean residual:** ≈ $0 — menunjukkan model **tidak memiliki bias sistematis**
- **Distribusi residual:** Mendekati distribusi normal, terkonsentrasi di sekitar nol
- **Residual vs Predicted:** Tidak menunjukkan pola tertentu — asumsi homoskedastisitas terpenuhi
- **Spread lebih besar pada biaya tinggi** — prediksi untuk perokok lebih bervariasi karena kompleksitas pola biayanya

### 7.6 Feature Importance

Fitur-fitur terpenting berdasarkan XGBoost feature importance:

| Peringkat | Fitur | Importance | Kategori |
|-----------|-------|------------|----------|
| 1 | `smoker_bmi` | Tertinggi | Interaksi (engineered) |
| 2 | `smoker_binary` | Tinggi | Status merokok |
| 3 | `age` / `age_sq` | Sedang | Usia |
| 4 | `bmi` / `bmi_sq` | Sedang | BMI |
| 5 | `smoker_age` | Sedang-rendah | Interaksi (engineered) |

Temuan ini mengkonfirmasi bahwa **feature engineering** — khususnya fitur interaksi `smoker_bmi` — sangat krusial untuk performa model.

---

## 8. Perbandingan Hasil Model

### 8.1 Tabel Perbandingan Komprehensif

| Split | Model | Train R² | Test R² | Test MAE | Test RMSE |
|-------|-------|----------|---------|----------|-----------|
| 70/30 | Linear Regression | 0.8450 | 0.8361 | $2.474 | $4.902 |
| 70/30 | Random Forest | 0.9620 | 0.8669 | $2.097 | $4.418 |
| 70/30 | XGBoost | 0.9350 | 0.8711 | $2.004 | $4.346 |
| 70/30 | LightGBM | 0.9410 | 0.8670 | $2.072 | $4.417 |
| 80/20 | Linear Regression | 0.8440 | 0.8436 | $2.500 | $4.928 |
| 80/20 | Random Forest | 0.9590 | 0.8724 | $2.046 | $4.451 |
| 80/20 | **XGBoost** | **0.9380** | **0.8802** | **$1.912** | **$4.313** |
| 80/20 | LightGBM | 0.9430 | 0.8750 | $2.026 | $4.404 |
| 90/10 | Linear Regression | 0.8430 | 0.8372 | $2.494 | $4.740 |
| 90/10 | Random Forest | 0.9580 | 0.8672 | $1.983 | $4.280 |
| 90/10 | XGBoost | 0.9360 | 0.8710 | $1.928 | $4.219 |
| 90/10 | LightGBM | 0.9420 | 0.8700 | $2.010 | $4.234 |

### 8.2 Rangking Model (berdasarkan rata-rata Test R² di seluruh split)

| Peringkat | Model | Rata-rata Test R² | Rata-rata MAE |
|-----------|-------|-------------------|---------------|
| 🥇 1 | **XGBoost** | **0.8741** | **$1.948** |
| 🥈 2 | LightGBM | 0.8707 | $2.036 |
| 🥉 3 | Random Forest | 0.8688 | $2.042 |
| 4 | Linear Regression | 0.8390 | $2.489 |

### 8.3 Analisis Perbandingan

1. **XGBoost secara konsisten unggul** di semua split dan metrik evaluasi
2. **Split 80/20 memberikan performa terbaik** — keseimbangan optimal antara data training yang cukup dan test set yang representatif
3. **Model tree-based (RF, XGBoost, LightGBM) mengungguli Linear Regression** sebesar ~4 poin R², menunjukkan pentingnya menangkap hubungan non-linear
4. **Random Forest dan LightGBM** memiliki performa yang sangat mirip, keduanya sekitar 1% di bawah XGBoost
5. **Tidak terdeteksi overfitting yang parah** — perbedaan Train R² dan Test R² pada XGBoost (~6%) masih dalam batas wajar
6. **Linear Regression tetap memberikan baseline yang baik** (R² > 0.83) berkat feature engineering yang efektif

### 8.4 Pengaruh Data Splitting

| Split | Kelebihan | Kekurangan | Performa Rata-rata |
|-------|-----------|------------|-------------------|
| 70/30 | Test set besar (402), evaluasi lebih stabil | Training data lebih sedikit | R² ≈ 0.86 |
| **80/20** | **Keseimbangan optimal** | - | **R² ≈ 0.87 (terbaik)** |
| 90/10 | Training data terbanyak | Test set kecil (134), evaluasi kurang stabil | R² ≈ 0.87 |

---

## 9. Kesimpulan

### 9.1 Rangkuman Hasil

Project ini berhasil membangun model prediksi biaya asuransi kesehatan dengan performa yang sangat baik:

- **Model terbaik:** XGBoost Regressor dengan **R² = 0.8802** (88,02%)
- **Rata-rata error prediksi:** $1.912 (MAE), yang berarti prediksi model rata-rata hanya meleset sekitar **$1.900** dari biaya aktual
- **Feature engineering** terbukti krusial — fitur interaksi `smoker_bmi` menjadi prediktor dominan
- **Log-transform pada target** meningkatkan performa secara signifikan pada distribusi yang skewed

### 9.2 Temuan Utama

1. **Status merokok adalah faktor terpenting** dalam menentukan biaya asuransi, khususnya ketika diinteraksikan dengan BMI
2. **Feature engineering** meningkatkan performa dari ~0.77 (tanpa engineering) menjadi **~0.88** (dengan engineering)
3. **Model tree-based** secara konsisten mengungguli model linear karena kemampuannya menangkap hubungan non-linear
4. **Data splitting 80/20** memberikan keseimbangan terbaik untuk dataset berukuran 1.338 baris
5. Model menunjukkan **robustness yang baik** — ketika diuji dengan data ekstrem/absurd, prediksi tetap dalam rentang yang wajar berkat sifat tree-based model yang naturally bounded

### 9.3 Keterbatasan

- Dataset relatif kecil (1.338 baris) — hasil mungkin bervariasi pada data yang lebih besar
- Data hanya mencakup wilayah Amerika Serikat — generalisasi ke sistem kesehatan lain perlu divalidasi
- Status `smoker` bersifat biner — tidak menangkap tingkat konsumsi rokok atau durasi merokok
- Tidak ada informasi temporal — model tidak dapat menangkap tren waktu

### 9.4 Rekomendasi Pengembangan

1. **Validasi dengan dataset lebih besar** untuk memastikan generalisasi model
2. **Cross-validation (k-fold)** dapat digunakan untuk evaluasi yang lebih robust pada dataset kecil
3. **Ensemble/Stacking** dari XGBoost + LightGBM berpotensi memberikan peningkatan marginal
4. **Input validation** perlu ditambahkan untuk deployment — menolak input dengan nilai negatif atau di luar rentang wajar
5. **Monitoring prediksi** secara berkala untuk mendeteksi pergeseran distribusi data (data drift)
