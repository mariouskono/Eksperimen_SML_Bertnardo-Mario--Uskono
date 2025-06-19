# -*- coding: utf-8 -*-
"""
Eksperimen Bertnardo Mario Uskono

Skrip ini melakukan analisis dan preprocessing data tempat wisata Bali.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Menentukan path relatif ke file dataset
dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset_tempat_wisata_bali-raw.csv')

# Memeriksa apakah file dataset ada
if not os.path.isfile(dataset_path):
    raise FileNotFoundError(f"File dataset tidak ditemukan di path: {dataset_path}")

# Memuat dataset
df = pd.read_csv(dataset_path)

# Menampilkan informasi dasar tentang dataset
print("Informasi Dataset:")
print(df.info())

# Menampilkan statistik deskriptif
print("\nStatistik Deskriptif:")
print(df.describe())

# Menampilkan jumlah nilai yang hilang per kolom
print("\nJumlah Nilai Hilang per Kolom:")
print(df.isnull().sum())

# Visualisasi distribusi rating
plt.figure(figsize=(10, 6))
sns.histplot(df['rating'], bins=20, kde=True)
plt.title('Distribusi Rating Tempat Wisata')
plt.xlabel('Rating')
plt.ylabel('Frekuensi')
plt.show()

# Visualisasi distribusi kategori
plt.figure(figsize=(10, 6))
sns.countplot(y='kategori', data=df)
plt.title('Distribusi Kategori Tempat Wisata')
plt.xlabel('Jumlah Tempat Wisata')
plt.ylabel('Kategori')
plt.show()

# Visualisasi korelasi antar fitur numerik
corr = df[['rating', 'latitude', 'longitude']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasi antara Fitur Numerik')
plt.show()

# Menangani missing values dengan mengisi nilai hilang pada kolom 'rating' dengan rata-rata
df['rating'].fillna(df['rating'].mean(), inplace=True)

# Menghapus duplikat
df.drop_duplicates(inplace=True)

# Standarisasi fitur numerik
scaler = StandardScaler()
df[['rating', 'latitude', 'longitude']] = scaler.fit_transform(df[['rating', 'latitude', 'longitude']])

# Encoding data kategorikal
encoder = LabelEncoder()
df['kategori'] = encoder.fit_transform(df['kategori'])
df['preferensi'] = encoder.fit_transform(df['preferensi'])

# Membagi data menjadi fitur dan target
X = df[['latitude', 'longitude', 'kategori', 'preferensi']]
y = df['rating']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Melakukan prediksi
y_pred = model.predict(X_test)

# Menampilkan hasil evaluasi model
print("\nEvaluasi Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Menyimpan dataset yang telah diproses ke file CSV
processed_file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset_tempat_wisata_bali_processed.csv')
df.to_csv(processed_file_path, index=False)
print(f"\nDataset yang telah diproses disimpan di: {processed_file_path}")
