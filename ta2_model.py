from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def train_regression_model(df):
    X = df[["Hours_Studied"]]
    y = df["Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Parameter Model ---")
    print("Koefisien (Slope):", model.coef_[0])
    print("Intercept:", model.intercept_)
    print("\n--- Evaluasi Model ---")
    print("R-squared:", round(r2, 3))
    print("RMSE:", round(rmse, 3))

    # Visualisasi hasil
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=X_test["Hours_Studied"], y=y_test, label="Data Aktual")
    sns.lineplot(x=X_test["Hours_Studied"], y=y_pred, color="red", label="Garis Regresi")
    plt.title("Prediksi Nilai Ujian Berdasarkan Jam Belajar")
    plt.xlabel("Jam Belajar")
    plt.ylabel("Nilai Ujian")
    plt.legend()
    plt.show()

    return model
def save_model(model, filename="model_regresi.pkl"):
    """Menyimpan model ke file"""
    joblib.dump(model, filename)
    print(f"\nModel berhasil disimpan ke file: {filename}")

def load_model(filename="model_regresi.pkl"):
    """Memuat model dari file"""
    model = joblib.load(filename)
    print(f"Model berhasil dimuat dari file: {filename}")
    return model
