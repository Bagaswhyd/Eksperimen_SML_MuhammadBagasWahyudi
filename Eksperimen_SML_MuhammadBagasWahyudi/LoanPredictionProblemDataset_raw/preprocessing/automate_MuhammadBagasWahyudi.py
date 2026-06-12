import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(raw_path, output_path):
    """
    Fungsi untuk melakukan preprocessing data secara otomatis.
    """
    print("Memulai preprocessing otomatis...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(raw_path)
        print(f"Data berhasil dimuat: {df.shape}")
    except FileNotFoundError:
        print("File dataset tidak ditemukan!")
        return

    # 2. Handling Missing Values
    num_cols = ['LoanAmount', 'Loan_Amount_Term']
    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # 3. Encoding
    le = LabelEncoder()
    cols_to_encode = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    for col in cols_to_encode:
        df[col] = le.fit_transform(df[col])

    # 4. Splitting & Scaling
    X = df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = df['Loan_Status']

    scaler = StandardScaler()
    scaled_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    X[scaled_cols] = scaler.fit_transform(X[scaled_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Save Data
    os.makedirs(output_path, exist_ok=True)
    X_train.to_csv(os.path.join(output_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_path, 'y_test.csv'), index=False)
    
    print(f"Preprocessing selesai. Data disimpan di: {output_path}")

if __name__ == "__main__":
    # Path disesuaikan dengan struktur folder
    # Karena file ini ada di folder 'preprocessing', path raw naik 1 level
    RAW_DATA = '../namadataset_raw/loan_data_raw.csv'
    OUTPUT_DIR = 'namadataset_preprocessing'
    
    preprocess_data(RAW_DATA, OUTPUT_DIR)