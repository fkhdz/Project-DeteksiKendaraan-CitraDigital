import os
import cv2
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import easyocr
import tkinter as tk
from tkinter import filedialog
from collections import Counter
import glob

# 1. KONFIGURASI PATH MODEL (LOKAL)
plat_model_path = "plat_nomor_yolov8s_best.pt"
jenis_model_path = "jenis_kendaraan_yolo_best.pt"
warna_model_path = "warna_kendaraan_resnet50.keras"
warna_class_path = "warna_kendaraan_classes.json"

print("Memuat model... (ini mungkin memakan waktu beberapa detik)")

detector_plat = YOLO(plat_model_path)
model_jenis = YOLO(jenis_model_path)
color_model = tf.keras.models.load_model(warna_model_path)
reader = easyocr.Reader(["en"], gpu=True) # Ubah ke True jika komputermu punya GPU NVIDIA

with open(warna_class_path, "r") as f:
    color_class_names = json.load(f)

print("Semua model berhasil dimuat!")

# 2. FUNGSI PEMROSESAN (Diambil dari Colab)
def bersihkan_teks_plat(text):
    if text is None: return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def deteksi_wilayah_plat(teks_plat):
    teks_plat = bersihkan_teks_plat(teks_plat)
    if not teks_plat: return "-", "Wilayah tidak dikenali"
    wilayah_map = {
        "AB": "Yogyakarta", "AD": "Surakarta / Solo", "AA": "Kedu",
        "AE": "Madiun", "AG": "Kediri", "A": "Banten", "B": "Jakarta",
        "D": "Bandung", "E": "Cirebon", "F": "Bogor", "G": "Pekalongan",
        "H": "Semarang", "K": "Pati", "L": "Surabaya", "M": "Madura",
        "N": "Malang", "P": "Besuki", "R": "Banyumas", "S": "Bojonegoro",
        "T": "Karawang", "W": "Sidoarjo", "Z": "Tasikmalaya"
    }
    if teks_plat[:2] in wilayah_map: return teks_plat[:2], wilayah_map[teks_plat[:2]]
    if teks_plat[:1] in wilayah_map: return teks_plat[:1], wilayah_map[teks_plat[:1]]
    return "-", "Wilayah tidak dikenali"

def baca_plat_dari_crop(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0: return "", 0.0
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    hasil_ocr = reader.readtext(crop_rgb)
    kandidat = [(bersihkan_teks_plat(item[1]), float(item[2])) for item in hasil_ocr if len(bersihkan_teks_plat(item[1])) >= 3]
    if not kandidat: return "", 0.0
    return sorted(kandidat, key=lambda x: x[1], reverse=True)[0]

def get_color_name(idx):
    if isinstance(color_class_names, list): return color_class_names[idx]
    if isinstance(color_class_names, dict):
        if str(idx) in color_class_names: return color_class_names[str(idx)]
        inverse = {v: k for k, v in color_class_names.items()}
        if idx in inverse: return inverse[idx]
    return f"Class-{idx}"

def normalisasi_nama_warna(warna):
    mapping = {"black": "Hitam", "white": "Putih", "grey": "Abu-abu", "gray": "Abu-abu", "silver": "Abu-abu", "red": "Merah", "blue": "Biru", "green": "Hijau", "yellow": "Kuning", "brown": "Coklat", "orange": "Oranye"}
    return mapping.get(str(warna).lower(), str(warna).capitalize())

def crop_area_bodi_kendaraan(crop_rgb, jenis=None):
    h, w = crop_rgb.shape[:2]
    if h <= 10 or w <= 10: return crop_rgb
    if jenis and "motor" in str(jenis).lower():
        return crop_rgb[int(h * 0.15):int(h * 0.88), int(w * 0.10):int(w * 0.90)]
    return crop_rgb[int(h * 0.32):int(h * 0.82), int(w * 0.15):int(w * 0.85)]

def prediksi_warna_crop(crop_rgb, image_size=(224, 224), top_k=3, jenis=None):
    crop_body = crop_area_bodi_kendaraan(crop_rgb, jenis=jenis)
    img = Image.fromarray(crop_body).convert("RGB").resize(image_size)
    img_array = np.expand_dims(np.array(img).astype("float32") / 255.0, axis=0)
    pred = color_model.predict(img_array, verbose=0)[0]
    top_idx = np.argsort(pred)[-top_k:][::-1]
    
    top_results = [{"warna": normalisasi_nama_warna(get_color_name(int(idx))), "confidence": float(pred[idx])} for idx in top_idx]
    return top_results[0]["warna"], top_results[0]["confidence"], top_results

def cari_plat_terdekat(kendaraan_box, daftar_plat, img_h, img_w):
    if not daftar_plat: return None
    vx1, vy1, vx2, vy2 = kendaraan_box
    expanded_box = (max(0, vx1 - 0.15*(vx2-vx1)), max(0, vy1 - 0.10*(vy2-vy1)), min(img_w, vx2 + 0.15*(vx2-vx1)), min(img_h, vy2 + 0.70*(vy2-vy1)))
    kandidat = []
    
    for plat in daftar_plat:
        px1, py1, px2, py2 = plat["bbox"]
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        inside = (expanded_box[0] <= pcx <= expanded_box[2] and expanded_box[1] <= pcy <= expanded_box[3])
        overlap_ratio = max(0, min(vx2, px2) - max(vx1, px1)) / max(1, px2 - px1)
        distance = ((pcx - (vx1+vx2)/2)**2 + (pcy - (vy1+vy2)/2)**2)**0.5
        score = distance * (0.5 if inside else 1) * (0.7 if overlap_ratio > 0.3 else 1) * (0.8 if pcy >= vy1 else 1)
        if inside or overlap_ratio > 0.25: kandidat.append((score, plat))
        
    return sorted(kandidat, key=lambda x: x[0])[0][1] if kandidat else None

# ==========================================
# 3. PIPELINE UTAMA DETEKSI (VERSI BATCH)
# ==========================================
def deteksi_gabungan_full(image_path, folder_output, conf_jenis=0.30, conf_plat=0.20):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Gambar gagal dibaca: {image_path}")
        return []
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_vis = img_rgb.copy()
    h, w = img_rgb.shape[:2]

    # Prediksi Jenis & Plat
    hasil_jenis = model_jenis.predict(source=image_path, conf=conf_jenis, verbose=False)
    boxes_kendaraan = hasil_jenis[0].boxes
    hasil_plat = detector_plat.predict(source=image_path, conf=conf_plat, imgsz=1280, verbose=False)
    boxes_plat = hasil_plat[0].boxes

    daftar_plat = []
    for pbox in boxes_plat:
        px1, py1, px2, py2 = map(int, pbox.xyxy[0])
        px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)
        pad_x, pad_y = int((px2 - px1) * 0.20), int((py2 - py1) * 0.35)
        px1_pad, py1_pad = max(0, px1 - pad_x), max(0, py1 - pad_y)
        px2_pad, py2_pad = min(w, px2 + pad_x), min(h, py2 + pad_y)
        
        crop_plat_bgr = img_bgr[py1_pad:py2_pad, px1_pad:px2_pad]
        teks_plat, conf_ocr = baca_plat_dari_crop(crop_plat_bgr)
        wilayah = deteksi_wilayah_plat(teks_plat)
        daftar_plat.append({
            "bbox": (px1, py1, px2, py2), "teks": teks_plat or "Tidak terbaca",
            "wilayah": wilayah, "conf_deteksi": float(pbox.conf[0]), "conf_ocr": conf_ocr
        })

    hasil_data = []
    if len(boxes_kendaraan) == 0:
        print("⚠️ Tidak ada kendaraan terdeteksi")
        return []

    for i, box in enumerate(boxes_kendaraan, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        jenis = model_jenis.names[int(box.cls[0])]
        
        crop_kendaraan_rgb = img_rgb[y1:y2, x1:x2]
        warna, conf_warna, top_warna = ("Tidak terdeteksi", 0.0, []) if crop_kendaraan_rgb.size == 0 else prediksi_warna_crop(crop_kendaraan_rgb, jenis=jenis)
        
        plat_terkait = cari_plat_terdekat((x1, y1, x2, y2), daftar_plat, h, w)
        if not plat_terkait and len(boxes_kendaraan) == 1 and len(daftar_plat) == 1: 
            plat_terkait = daftar_plat[0]
        
        nomor_plat = plat_terkait["teks"] if plat_terkait else "Tidak terdeteksi"
        wilayah_plat = plat_terkait["wilayah"] if plat_terkait else "-"

        # Drawing Bounding Boxes
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{jenis} | {warna}"
        cv2.putText(img_vis, label, (x1 + 6, max(10, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        
        if plat_terkait:
            px1, py1, px2, py2 = plat_terkait["bbox"]
            cv2.rectangle(img_vis, (px1, py1), (px2, py2), (255, 255, 0), 2)
            cv2.putText(img_vis, nomor_plat, (px1 + 5, max(10, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Simpan nama file asli ke dalam tabel
        nama_file = os.path.basename(image_path)
        hasil_data.append({
            "File": nama_file, "Kendaraan": i, "Jenis": jenis, 
            "Warna": warna, "Nomor Plat": nomor_plat, "Wilayah": wilayah_plat[1]
        })

    # Simpan Gambar Hasil (Otomatis tanpa menahan script)
    os.makedirs(folder_output, exist_ok=True)
    path_simpan = os.path.join(folder_output, f"hasil_{os.path.basename(image_path)}")
    img_bgr_out = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR) # OpenCV butuh format BGR untuk menyimpan
    cv2.imwrite(path_simpan, img_bgr_out)

    return hasil_data

# ==========================================
# 4. EKSEKUSI DATASET (BATCH PROCESSING)
# ==========================================
if __name__ == "__main__":
    # Tentukan nama folder berisi dataset gambar
    folder_dataset = "dataset_kendaraan" 
    folder_output = "hasil_deteksi_batch"

    print(f"Mencari gambar di folder '{folder_dataset}'...")
    
    # Ambil semua gambar JPG dan PNG dari folder dataset
    ekstensi = ('*.jpg', '*.jpeg', '*.png')
    daftar_gambar = []
    for eks in ekstensi:
        daftar_gambar.extend(glob.glob(os.path.join(folder_dataset, eks)))

    if not daftar_gambar:
        print(f"❌ Tidak ada gambar ditemukan. Pastikan folder '{folder_dataset}' ada dan berisi gambar.")
    else:
        print(f"✅ Ditemukan {len(daftar_gambar)} gambar. Memulai proses batch...\n")
        semua_hasil = []

        for path_gambar in daftar_gambar:
            print(f"Sedang memproses: {os.path.basename(path_gambar)}")
            hasil = deteksi_gabungan_full(path_gambar, folder_output)
            if hasil:
                semua_hasil.extend(hasil)

        # Ekspor gabungan data prediksi ke CSV agar mudah diolah seperti log
        if semua_hasil:
            df_total = pd.DataFrame(semua_hasil)
            file_csv = os.path.join(folder_output, "rekap_prediksi.csv")
            df_total.to_csv(file_csv, index=False)
            print(f"\n🎉 Proses Selesai!")
            print(f"Gambar beranotasi dan 'rekap_prediksi.csv' tersimpan di folder: {folder_output}/")

# 3. PIPELINE UTAMA DETEKSI (VERSI BATCH)
def deteksi_gabungan_full(image_path, folder_output, conf_jenis=0.30, conf_plat=0.20):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Gambar gagal dibaca: {image_path}")
        return []
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_vis = img_rgb.copy()
    h, w = img_rgb.shape[:2]

    hasil_jenis = model_jenis.predict(source=image_path, conf=conf_jenis, verbose=False)
    boxes_kendaraan = hasil_jenis[0].boxes
    hasil_plat = detector_plat.predict(source=image_path, conf=conf_plat, imgsz=1280, verbose=False)
    boxes_plat = hasil_plat[0].boxes

    daftar_plat = []
    for pbox in boxes_plat:
        px1, py1, px2, py2 = map(int, pbox.xyxy[0])
        px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)
        pad_x, pad_y = int((px2 - px1) * 0.20), int((py2 - py1) * 0.35)
        px1_pad, py1_pad = max(0, px1 - pad_x), max(0, py1 - pad_y)
        px2_pad, py2_pad = min(w, px2 + pad_x), min(h, py2 + pad_y)
        
        crop_plat_bgr = img_bgr[py1_pad:py2_pad, px1_pad:px2_pad]
        teks_plat, conf_ocr = baca_plat_dari_crop(crop_plat_bgr)
        wilayah = deteksi_wilayah_plat(teks_plat)
        daftar_plat.append({
            "bbox": (px1, py1, px2, py2), "teks": teks_plat or "Tidak terbaca",
            "wilayah": wilayah, "conf_deteksi": float(pbox.conf[0]), "conf_ocr": conf_ocr
        })

    hasil_data = []
    if len(boxes_kendaraan) == 0:
        print("⚠️ Tidak ada kendaraan terdeteksi")
        return []

    for i, box in enumerate(boxes_kendaraan, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        jenis = model_jenis.names[int(box.cls[0])]
        
        crop_kendaraan_rgb = img_rgb[y1:y2, x1:x2]
        warna, conf_warna, top_warna = ("Tidak terdeteksi", 0.0, []) if crop_kendaraan_rgb.size == 0 else prediksi_warna_crop(crop_kendaraan_rgb, jenis=jenis)
        
        plat_terkait = cari_plat_terdekat((x1, y1, x2, y2), daftar_plat, h, w)
        if not plat_terkait and len(boxes_kendaraan) == 1 and len(daftar_plat) == 1: 
            plat_terkait = daftar_plat[0]
        
        nomor_plat = plat_terkait["teks"] if plat_terkait else "Tidak terdeteksi"
        wilayah_plat = plat_terkait["wilayah"] if plat_terkait else "-"

        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{jenis} | {warna}"
        cv2.putText(img_vis, label, (x1 + 6, max(10, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        
        if plat_terkait:
            px1, py1, px2, py2 = plat_terkait["bbox"]
            cv2.rectangle(img_vis, (px1, py1), (px2, py2), (255, 255, 0), 2)
            cv2.putText(img_vis, nomor_plat, (px1 + 5, max(10, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        nama_file = os.path.basename(image_path)
        hasil_data.append({
            "File": nama_file, "Kendaraan": i, "Jenis": jenis, 
            "Warna": warna, "Nomor Plat": nomor_plat, "Wilayah": wilayah_plat[1]
        })

    os.makedirs(folder_output, exist_ok=True)
    path_simpan = os.path.join(folder_output, f"hasil_{os.path.basename(image_path)}")
    img_bgr_out = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR) 
    cv2.imwrite(path_simpan, img_bgr_out)

    return hasil_data


# 4. EKSEKUSI DATASET (BATCH PROCESSING)
if __name__ == "__main__":
    folder_dataset = "dataset_kendaraan" 
    folder_output = "hasil_deteksi_batch"

    print(f"Mencari gambar di folder '{folder_dataset}'...")
    
    ekstensi = ('*.jpg', '*.jpeg', '*.png')
    daftar_gambar = []
    for eks in ekstensi:
        daftar_gambar.extend(glob.glob(os.path.join(folder_dataset, eks)))

    if not daftar_gambar:
        print(f"❌ Tidak ada gambar ditemukan. Pastikan folder '{folder_dataset}' ada dan berisi gambar.")
    else:
        print(f"✅ Ditemukan {len(daftar_gambar)} gambar. Memulai proses batch...\n")
        semua_hasil = []

        for path_gambar in daftar_gambar:
            print(f"Sedang memproses: {os.path.basename(path_gambar)}")
            hasil = deteksi_gabungan_full(path_gambar, folder_output)
            if hasil:
                semua_hasil.extend(hasil)
                
        if semua_hasil:
            df_total = pd.DataFrame(semua_hasil)
            file_csv = os.path.join(folder_output, "rekap_prediksi.csv")
            df_total.to_csv(file_csv, index=False)
            print(f"\n🎉 Proses Selesai!")
            print(f"Gambar beranotasi dan 'rekap_prediksi.csv' tersimpan di folder: {folder_output}/")