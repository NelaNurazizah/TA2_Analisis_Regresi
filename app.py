from flask import Flask, render_template, request, url_for
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Muat model
model = joblib.load("model_regresi.pkl")

# Dataset (untuk gambar scatter dan garis regresi)
df = pd.read_csv("study_hours.csv")

def buat_grafik(prediksi_x=None, prediksi_y=None):
    """Membuat grafik regresi dan menyimpannya sebagai file PNG"""
    X = df["Hours_Studied"].values.reshape(-1, 1)
    y = df["Marks"].values
    y_pred = model.predict(X)

    plt.figure(figsize=(6,4))
    plt.scatter(X, y, label="Data Aktual")
    plt.plot(X, y_pred, color="red", label="Garis Regresi")

    # Tambahkan titik prediksi jika ada
    if prediksi_x is not None and prediksi_y is not None:
        plt.scatter(prediksi_x, prediksi_y, color="green", s=80, label="Prediksi Baru")

    plt.title("Hubungan Jam Belajar dengan Nilai Ujian")
    plt.xlabel("Jam Belajar")
    plt.ylabel("Nilai Ujian")
    plt.legend()

    grafik_path = os.path.join("static", "grafik.png")
    plt.savefig(grafik_path)
    plt.close()

@app.route("/")
def home():
    # Buat grafik saat halaman dimuat
    buat_grafik()
    grafik_url = url_for("static", filename="grafik.png")
    return render_template("index.html", grafik_url=grafik_url)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        jam_belajar = float(request.form["jam_belajar"])
        prediksi = model.predict(np.array([[jam_belajar]]))
        hasil = round(prediksi[0], 2)

        # buat grafik dengan titik prediksi
        buat_grafik(prediksi_x=jam_belajar, prediksi_y=hasil)
        grafik_url = url_for("static", filename="grafik.png")

        return render_template("index.html", prediksi=hasil, jam=jam_belajar, grafik_url=grafik_url)
    except:
        return render_template("index.html", error="Input tidak valid")

if __name__ == "__main__":
    app.run(debug=True)
