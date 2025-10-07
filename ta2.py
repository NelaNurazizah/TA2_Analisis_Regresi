import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cek folder kerja saat ini
print("Folder kerja:", os.getcwd())

# Ganti dengan nama file dataset kamu
df = pd.read_csv("study_hours.csv")

# Tampilkan data
print("\n--- 5 Baris Pertama ---")
print(df.head())

print("\n--- Nama Kolom ---")
print(df.columns)

print("\n--- Info Dataset ---")
print(df.info())

print("\n--- Statistik Deskriptif ---")
print(df.describe())

# Visualisasi hubungan antara jam belajar dan nilai
plt.figure(figsize=(6,4))
sns.regplot(x="Hours_Studied", y="Marks", data=df, ci=None)
plt.title("Hubungan Jam Belajar dengan Nilai Ujian")
plt.xlabel("Jam Belajar")
plt.ylabel("Nilai Ujian")
plt.show()
