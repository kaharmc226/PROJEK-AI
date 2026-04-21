# 🏥 Prediksi Biaya Asuransi Kesehatan

Aplikasi prediksi biaya asuransi kesehatan berbasis machine learning yang dibangun menggunakan **XGBoost** dengan rekayasa fitur tingkat lanjut. Proyek ini mencakup seluruh pipeline dari eksplorasi data hingga model yang di-*deploy* sebagai layanan web.

---

## 📋 Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Dataset](#dataset)
- [Struktur Proyek](#struktur-proyek)
- [Pipeline Machine Learning](#pipeline-machine-learning)
- [Rekayasa Fitur](#rekayasa-fitur)
- [Hasil Model](#hasil-model)
- [Deployment Web](#deployment-web)
- [Cara Mengakses Aplikasi](#cara-mengakses-aplikasi)
- [Menjalankan Secara Lokal](#menjalankan-secara-lokal)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)

---

## Gambaran Umum

Proyek ini bertujuan membangun model regresi yang akurat untuk memprediksi biaya asuransi kesehatan individu berdasarkan data demografis dan gaya hidup. Model terbaik kemudian di-*deploy* sebagai aplikasi web interaktif yang dapat diakses secara publik melalui platform Render.

Pendekatan utama yang digunakan:
- Rekayasa fitur intensif, terutama interaksi variabel **perokok × BMI** sebagai prediktor dominan biaya
- Pelatihan model dengan akselerasi GPU (XGBoost, LightGBM, Random Forest)
- Transformasi target `log1p(charges)` untuk menangani distribusi right-skewed
- Validasi input berbasis rentang data latih

---

## Dataset

| Properti | Detail |
|---|---|
| **Nama** | Medical Cost Personal Dataset |
| **File** | `insurance.csv` |
| **Jumlah Baris** | 1.338 |
| **Jumlah Kolom** | 7 |

### Kolom Dataset

| Kolom | Tipe | Keterangan |
|---|---|---|
| `age` | Integer | Usia pemegang polis (18 – 64) |
| `sex` | Kategorikal | Jenis kelamin (`male` / `female`) |
| `bmi` | Float | Body Mass Index (15.96 – 53.13) |
| `children` | Integer | Jumlah anak tanggungan (0 – 5) |
| `smoker` | Kategorikal | Status merokok (`yes` / `no`) |
| `region` | Kategorikal | Wilayah AS (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges` | Float | **Target** — Biaya asuransi dalam USD |

---

## Struktur Proyek

```
classic-insurance/
│
├── insurance.csv                  # Dataset utama
├── best_model_classic.pkl         # Model terbaik yang tersimpan (XGBoost)
│
├── insurance_tuned_gpu.ipynb      # Notebook utama: EDA, pelatihan, & tuning model
├── custom_inference.ipynb         # Notebook inferensi & stress-test prediksi
├── insurance_inference.ipynb      # Notebook evaluasi inferensi tambahan
│
├── data_sparsity_analysis.png     # Visualisasi analisis sparsity data
│
└── render-app/                    # Aplikasi web untuk deployment
    ├── app.py                     # Flask API backend
    ├── requirements.txt           # Dependensi Python
    ├── render.yaml                # Konfigurasi deployment Render
    ├── model/
    │   └── best_model_classic.pkl # Salinan model untuk server
    └── static/
        ├── index.html             # Antarmuka pengguna (UI)
        ├── style.css              # Styling glassmorphism dark-mode
        └── script.js             # Logika client-side & pemanggilan API
```

---

## Pipeline Machine Learning

```
Dataset (insurance.csv)
        │
        ▼
  Eksplorasi Data (EDA)
  - Distribusi variabel
  - Analisis korelasi
  - Deteksi outlier
        │
        ▼
  Pra-pemrosesan
  - Encoding kategorikal
  - Transformasi target: log1p(charges)
        │
        ▼
  Rekayasa Fitur (21 fitur total)
  - Fitur interaksi (smoker × BMI)
  - Fitur polinomial (age², bmi²)
  - Fitur biner (is_obese, has_children, dll.)
        │
        ▼
  Pelatihan & Tuning Model (GPU)
  - XGBoost Regressor
  - LightGBM
  - Random Forest
        │
        ▼
  Evaluasi & Seleksi Model Terbaik
  - Metrik: R², RMSE, MAE
        │
        ▼
  Simpan Model → best_model_classic.pkl
        │
        ▼
  Deployment → Render (Flask + Gunicorn)
```

---

## Rekayasa Fitur

Model menggunakan **21 fitur rekayasa** yang diturunkan dari 6 fitur asli:

| Fitur | Keterangan |
|---|---|
| `age` | Usia asli |
| `bmi` | BMI asli |
| `children` | Jumlah anak asli |
| `smoker_binary` | Status perokok (0/1) |
| `smoker_bmi` | Interaksi: perokok × BMI ⭐ |
| `smoker_age` | Interaksi: perokok × usia |
| `age_sq` | Usia kuadrat |
| `bmi_sq` | BMI kuadrat |
| `age_bmi` | Interaksi: usia × BMI |
| `is_obese` | BMI ≥ 30 (0/1) |
| `is_overweight` | BMI ≥ 25 (0/1) |
| `smoker_obese` | Interaksi: perokok × obesitas |
| `age_group_young` | Usia < 30 (0/1) |
| `age_group_mid` | 30 ≤ usia < 50 (0/1) |
| `age_group_senior` | Usia ≥ 50 (0/1) |
| `has_children` | Memiliki anak (0/1) |
| `log_bmi` | log1p(BMI) |
| `sex_male` | Jenis kelamin laki-laki (0/1) |
| `region_northwest` | Wilayah barat laut (0/1) |
| `region_southeast` | Wilayah tenggara (0/1) |
| `region_southwest` | Wilayah barat daya (0/1) |

> ⭐ Fitur `smoker_bmi` adalah prediktor terkuat — perokok dengan BMI tinggi memiliki biaya asuransi yang jauh lebih besar.

---

## Hasil Model

| Model | R² | RMSE | Catatan |
|---|---|---|---|
| **XGBoost (terbaik)** | **~0.87** | Terbaik | Dipilih & disimpan |
| LightGBM | Kompetitif | — | Diuji dalam tuning |
| Random Forest | Lebih rendah | — | Baseline |

Model dilatih pada target `log1p(charges)` dan hasil prediksi dibalik dengan `expm1()` untuk mendapatkan nilai USD yang sesungguhnya.

---

## Deployment Web

Aplikasi web dibangun dengan arsitektur berikut:

```
Browser Pengguna
      │  HTTP Request
      ▼
  Flask App (app.py)
  ├── GET  /          → Melayani index.html (UI)
  └── POST /api/predict
       ├── Validasi input
       ├── Rekayasa fitur (21 fitur)
       ├── Prediksi model (XGBoost)
       └── Kembalikan JSON: { predicted_charges, features, warnings }
      │
      ▼
  Gunicorn (WSGI Server)
      │
      ▼
  Render.com (Cloud Platform)
  Region: Singapore | Plan: Free | Python 3.11.12
```

### Fitur Aplikasi Web
- 🎨 Antarmuka dark-mode dengan efek glassmorphism
- ✅ Validasi input real-time (rentang data latih)
- ⚠️ Peringatan otomatis untuk nilai di luar rentang
- 📊 Tabel rincian 21 fitur hasil rekayasa
- 🔢 Animasi counter untuk hasil prediksi
- 🌐 Antarmuka sepenuhnya dalam **Bahasa Indonesia**

---

## Cara Mengakses Aplikasi

Aplikasi telah di-*deploy* dan dapat diakses secara publik melalui tautan berikut:

> 🔗 **URL Aplikasi:** *(akan diisi)*

### Panduan Penggunaan

1. **Buka** tautan aplikasi di browser
2. **Isi formulir** dengan data pasien:
   - **Usia** — masukkan angka antara 18–64
   - **Jenis Kelamin** — pilih Laki-laki atau Perempuan
   - **BMI** — masukkan nilai antara 15.96–53.13
   - **Jumlah Anak** — masukkan angka antara 0–5
   - **Perokok** — pilih Ya atau Tidak
   - **Wilayah** — pilih salah satu dari empat wilayah
3. **Klik "Prediksi Biaya"** — hasil akan muncul di bawah formulir
4. **Lihat detail** fitur rekayasa dengan mengklik *"Lihat Fitur Rekayasa (21)"*

> ⚠️ Nilai di luar rentang data latih akan tetap diproses tetapi akan menampilkan **peringatan** karena akurasi prediksi mungkin berkurang.

---

## Menjalankan Secara Lokal

### Prasyarat
- Python 3.11+
- `pip`

### Langkah-langkah

```bash
# 1. Masuk ke direktori aplikasi
cd classic-insurance/render-app

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan server Flask
python app.py
```

Aplikasi akan berjalan di `http://localhost:5000`.

### Dependensi

| Library | Versi |
|---|---|
| Flask | 3.1.1 |
| Gunicorn | 23.0.0 |
| XGBoost | 3.2.0 |
| NumPy | 2.4.3 |
| Pandas | 2.2.3 |
| Joblib | 1.5.3 |

---

## Teknologi yang Digunakan

| Kategori | Teknologi |
|---|---|
| **Machine Learning** | XGBoost, LightGBM, Scikit-learn |
| **Bahasa Pemrograman** | Python 3.11 |
| **Backend** | Flask, Gunicorn |
| **Frontend** | HTML5, CSS3 (Glassmorphism), JavaScript (Vanilla) |
| **Deployment** | Render.com |
| **Notebook** | Jupyter Notebook |
| **Akselerasi** | GPU (pelatihan model) |

---

*Proyek ini dikembangkan sebagai studi kasus prediksi biaya asuransi kesehatan menggunakan teknik machine learning modern.*
