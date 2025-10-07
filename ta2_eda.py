import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    print("\n--- 5 Baris Pertama ---")
    print(df.head())
    print("\n--- Info Dataset ---")
    print(df.info())
    print("\n--- Statistik Deskriptif ---")
    print(df.describe())

    # Hapus baris kosong
    df = df.dropna(subset=["Hours_Studied", "Marks"])
    # Hapus duplikat
    df = df.drop_duplicates()
    
    print("\n--- Data Setelah Dibersihkan ---")
    print(df.shape)
    
    return df
