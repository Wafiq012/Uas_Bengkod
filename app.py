import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("🎯 Aplikasi Prediksi Churn Pelanggan")
st.write("UAS Bengkel Koding Data Science")

# --- PERBAIKAN DI SINI: Memuat model dengan pengaman try-except ---
model = None
try:
    model = joblib.load('best_churn_model_pipeline.pkl')
except Exception as e:
    st.error(f"Gagal memuat file model pkl: {e}")
    st.info("Pastikan file 'best_churn_model_pipeline.pkl' ada di repositori GitHub Anda.")

st.sidebar.header("📋 Input Fitur Karakteristik Pelanggan")

# Form Input Sederhana
gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.sidebar.number_input("Usia", min_value=18, max_value=100, value=30)
country = st.sidebar.text_input("Negara Asal", "United States")
city = st.sidebar.text_input("Kota", "New York")
is_premium_user = st.sidebar.selectbox("Premium User?", [0, 1])
total_visits = st.sidebar.number_input("Total Kunjungan", min_value=0, value=10)
total_spent = st.sidebar.number_input("Total Pengeluaran", min_value=0.0, value=50000.0)
satisfaction_score = st.sidebar.slider("Skor Kepuasan", 1.0, 5.0, 3.5)

# Jadikan input ke bentuk DataFrame
input_df = pd.DataFrame([{
    'gender': gender, 'age': age, 'country': country, 'city': city,
    'is_premium_user': is_premium_user, 'total visits': total_visits,
    'total_spent': total_spent, 'satisfaction score': satisfaction_score
}])

st.subheader("Data Input Pelanggan:")
st.dataframe(input_df)

if st.button("🚀 Prediksi Potensi Churn"):
    if model is None:
        st.error("Model tidak tersedia. Tidak dapat melakukan prediksi.")
    else:
        try:
            prediction = model.predict(input_df)
            if prediction[0] == 1:
                st.error("⚠️ PELANGGAN BERPOTENSI CHURN (BERHENTI BERLANGGANAN)")
            else:
                st.success("✅ PELANGGAN TETAP BERLANGGANAN (LOYAL)")
        except Exception as e:
            st.error(f"Terjadi kesalahan kesesuaian kolom: {e}")
            st.info("Tips: Pastikan semua kolom inputan sudah lengkap sesuai struktur dataset.")
