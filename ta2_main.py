from ta2_eda import load_and_clean_data
from ta2_model import train_regression_model, save_model, load_model

if __name__ == "__main__":
    # Path ke dataset
    path = "study_hours.csv"

    # Langkah 1: Load dan bersihkan data
    df = load_and_clean_data(path)

    # Langkah 2: Latih model regresi linear
    model = train_regression_model(df)

    # Langkah 3: Simpan model ke file
    save_model(model, "model_regresi.pkl")

    # (Opsional) Tes muat ulang model
    loaded_model = load_model("model_regresi.pkl")
