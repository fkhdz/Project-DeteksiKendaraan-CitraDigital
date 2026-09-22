# 🚗 Sistem Deteksi Plat Nomor, Jenis, & Warna Kendaraan

**Project Pengolahan Citra | Teknik Informatika UNNES 2026**

Dosen Pengampu: **Endang Sugiharti, S.Si., M.Kom.**

## 📌 Deskripsi Proyek

Proyek ini adalah sistem *Computer Vision* cerdas yang dirancang untuk menganalisis citra kendaraan di Indonesia secara komprehensif. Sistem ini menggabungkan beberapa model *Deep Learning* untuk melakukan tiga tugas utama secara bersamaan:

1. **Mendeteksi Jenis Kendaraan:** Mengenali apakah objek adalah mobil, motor, atau truk menggunakan model YOLO.

2. **Mengklasifikasi Warna Kendaraan:** Memprediksi warna dominan bodi kendaraan menggunakan model ResNet50.

3. **Mendeteksi & Membaca Plat Nomor:** Melokalisasi posisi plat nomor kendaraan Indonesia, mengekstrak teks menggunakan EasyOCR, dan memetakan kode wilayahnya.

## 📊 Dataset yang Digunakan

Sistem ini dilatih menggunakan dataset yang berfokus pada kondisi jalan dan kendaraan di Indonesia:

| No | Kegunaan | Nama Dataset | Sumber | 
 | ----- | ----- | ----- | ----- | 
| 1 | **Deteksi Plat Nomor** | Indonesia License Plate Computer Vision | [Roboflow Universe](https://universe.roboflow.com/alfian-fc0es/indonesia-license-plate/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true&utm_source=gemini) | 
| 2 | **Jenis Kendaraan** | Indonesia Vehicle Computer Vision Model | [Roboflow Universe](https://universe.roboflow.com/arkhsat-project/indonesia-vehicle-ww6hv?utm_source=gemini) | 
| 3 | **Warna Kendaraan** | Vehicle Color Classification Dataset | [HuggingFace](https://huggingface.co/datasets/WandererGuy/vehicle-color-classification-dataset?utm_source=gemini) | 


### Struktur Direktori

Pastikan kamu telah mengunduh model hasil *training* (karena file model terlalu besar untuk GitHub) dan meletakkannya di dalam folder root repositori agar sesuai dengan struktur berikut:

```text
Project-DeteksiKendaraan/
│
├── deteksi_lokal.py                 # Script utama untuk GUI / inferensi single image
├── deteksi_batch.py                 # Script untuk inferensi banyak gambar sekaligus
├── plat_nomor_yolov8s_best.pt       # Model YOLO Plat (Download terpisah)
├── jenis_kendaraan_yolo_best.pt     # Model YOLO Jenis Kendaraan (Download terpisah)
├── warna_kendaraan_resnet50.keras   # Model ResNet50 Warna (Download terpisah)
├── warna_kendaraan_classes.json     # File mapping class warna
├── dataset_kendaraan/               # Folder untuk menaruh gambar uji coba
└── README.md
```

## 🚀 Cara Penggunaan

Kamu dapat menjalankan proyek ini dalam dua mode:

### Mode 1: Single Image (GUI File Explorer)

Gunakan mode ini untuk memilih satu gambar lewat *pop-up window* dan langsung melihat hasil visualisasinya.

```bash
python deteksi_lokal.py
```

### Mode 2: Batch Processing (Banyak Gambar)

Gunakan mode ini jika kamu ingin memproses puluhan/ratusan gambar sekaligus yang ada di dalam folder `dataset_kendaraan/`. Hasilnya akan berupa CSV (`rekap_prediksi.csv`) dan gambar beranotasi.

```bash
python deteksi_batch.py
```

## 📝 Catatan Penting

* Model `.pt` dan `.keras` tidak disertakan di repositori ini jika ukurannya melebihi batas GitHub (100MB). Kamu harus menggunakan Git LFS atau menyediakan tautan Google Drive eksternal di sini agar pengguna lain bisa mengunduhnya.

* Penggunaan prosesor GPU (NVIDIA) sangat disarankan agar performa YOLO dan EasyOCR berjalan *real-time*. Jika menggunakan CPU, proses inferensi mungkin memakan waktu beberapa detik per gambar.