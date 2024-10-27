import numpy as np
import joblib
import streamlit as st

# Memuat model yang telah dilatih dengan penanganan kesalahan
model_filename = 'logistic_regression_model.pkl'
try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    st.error("Model tidak ditemukan. Pastikan file model ada di direktori yang benar.")
    st.stop()  # Hentikan eksekusi jika model tidak ditemukan

# Judul aplikasi web
st.title('Prediksi Penyakit Jantung')

# Membuat kolom untuk input data pengguna
col1, col2, col3 = st.columns(3)

with col1:
    age = st.text_input('Umur', '0')

with col2:
    sex = st.selectbox('Jenis Kelamin', ('0', '1'))  # 0 untuk Wanita, 1 untuk Pria

with col3:
    cp = st.selectbox('Jenis Nyeri Dada', ('0', '1', '2', '3'))

with col1:
    trestbps = st.text_input('Tekanan Darah', '0')

with col2:
    chol = st.text_input('Nilai Kolesterol', '0')

with col3:
    fbs = st.selectbox('Gula Darah > 120 mg/dl', ('0', '1'))  # 1 jika Benar, 0 jika Salah

with col1:
    restecg = st.selectbox('Hasil Elektrokardiografi', ('0', '1', '2'))

with col2:
    thalach = st.text_input('Detak Jantung Maksimum', '0')

with col3:
    exang = st.selectbox('Induksi Angina', ('0', '1'))  # 1 jika Benar, 0 jika Salah

with col1:
    oldpeak = st.text_input('ST Depression', '0')

with col2:
    slope = st.selectbox('Slope', ('0', '1', '2'))

with col3:
    ca = st.selectbox('Nilai CA', ('0', '1', '2', '3', '4'))

with col1:
    thal = st.selectbox('Nilai Thal', ('0', '1', '2', '3'))

# Variabel untuk menampilkan hasil prediksi
heart_diagnosis = ''

# Tombol untuk melakukan prediksi
if st.button('Prediksi Penyakit Jantung'):
    try:
        # Mengonversi input ke tipe float
        heart_prediction = [
            float(age), float(sex), float(cp), float(trestbps), float(chol), 
            float(fbs), float(restecg), float(thalach), float(exang), 
            float(oldpeak), float(slope), float(ca), float(thal)
        ]
        
        # Mengubah data input menjadi bentuk yang sesuai untuk prediksi tunggal
        input_data = np.asarray(heart_prediction).reshape(1, -1)

        # Melakukan prediksi menggunakan model yang telah dimuat
        prediction = model.predict(input_data)

        # Menentukan hasil prediksi
        if prediction[0] == 0:
            heart_diagnosis = 'Pasien Tidak Terkena Penyakit Jantung'
        else:
            heart_diagnosis = 'Pasien Terkena Penyakit Jantung'

        # Menampilkan hasil
        st.success(heart_diagnosis)
    except ValueError:
        st.error("Terjadi kesalahan dalam memasukkan data. Pastikan semua input sudah benar dan dalam format yang sesuai.")
