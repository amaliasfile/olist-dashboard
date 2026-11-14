# Proyek Analisis Data: E-Commerce Public Dataset (Olist)

Dashboard interaktif (Streamlit) untuk menjawab 3 pertanyaan bisnis:
1. Kategori produk dengan revenue tertinggi (Jan 2017 - Agu 2018).
2. Negara bagian dengan pelanggan terbanyak dan waktu pengiriman terlama (geospatial analysis).
3. Segmentasi pelanggan dengan RFM analysis.

## Struktur Direktori

```
submission
├───dashboard
│   ├───main_data.csv
│   └───dashboard.py
├───data
│   ├───customers_dataset.csv
│   ├───geolocation_dataset.csv
│   ├───order_items_dataset.csv
│   ├───orders_dataset.csv
│   ├───product_category_name_translation.csv
│   └───products_dataset.csv
├───notebook.ipynb
├───README.md
├───requirements.txt
└───url.txt
```

## Setup Environment - Anaconda

```
conda create --name main-ds python=3.11
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal

```
mkdir submission
cd submission
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Menjalankan Dashboard

Dari folder `submission`, jalankan:

```
streamlit run dashboard/dashboard.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`.

## Menjalankan Notebook

Buka `notebook.ipynb` dari folder `submission` (agar path `data/` terbaca), lalu jalankan semua cell. Cell terakhir pada tahap Cleaning akan menghasilkan ulang `dashboard/main_data.csv`.

## Tautan Dashboard

Lihat `url.txt`.
