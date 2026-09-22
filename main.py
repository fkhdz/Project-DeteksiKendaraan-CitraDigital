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

