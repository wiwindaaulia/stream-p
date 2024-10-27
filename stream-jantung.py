import streamlit as st
import numpy as np
import joblib

# Memuat model yang sudah disimpan
model = joblib.load('logistic_regression_model.pkl')

# Judul aplikasi
st.title("Prediksi Penyakit Jantung")

# Masukkan input pengguna
def user_input_features():
    age = st.number_input("Usia", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Jenis Kelamin (0: Wanita, 1: Pria)", options=[0, 1])
    cp = st.selectbox("Tipe Nyeri Dada (0-3)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Gula Darah Puasa (>120 mg/dl, 1: Ya, 0: Tidak)", options=[0, 1])
    restecg = st.selectbox("Hasil Elektrokardiografi (0-2)", options=[0, 1, 2])
    thalach = st.number_input("Denyut Jantung Maksimum", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Angina yang Direspon Olahraga (1: Ya, 0: Tidak)", options=[0, 1])
    oldpeak = st.number_input("ST Depresi Terinduksi", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    slope = st.selectbox("Kemiringan Puncak ST (0-2)", options=[0, 1, 2])
    ca = st.selectbox("Jumlah Pembuluh Besar (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (1: Normal; 2: Cacat Tetap; 3: Cacat Reversibel)", options=[1, 2, 3])

    # Membuat array input pengguna
    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    return features

input_data = user_input_features()

# Tombol untuk membuat prediksi
if st.button("Prediksi"):
    input_data_reshaped = input_data.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        st.success("Pasien Tidak Terkena Penyakit Jantung")
    else:
        st.error("Pasien Terkena Penyakit Jantung")
