import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("🎯 Aplikasi Prediksi Churn Pelanggan")
st.write("UAS Bengkel Koding Data Science")

# Memuat model pipeline yang sudah diunduh tadi
model = joblib.load('best_churn_model_pipeline.pkl')

st.sidebar.header("📋 Input Fitur Karakteristik Pelanggan")

# Form Input Sederhana (Sesuaikan atau lengkapi sesuai kolom dataset Anda)
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
    # Catatan: Jika saat running muncul error "ValueError", 
    # pastikan semua nama kolom di sini sama persis dengan dataset asli Anda!
}])

st.subheader("Data Input Pelanggan:")
st.dataframe(input_df)

if st.button("🚀 Prediksi Potensi Churn"):
    try:
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.error("⚠️ PELANGGAN BERPOTENSI CHURN (BERHENTI BERLANGGANAN)")
        else:
            st.success("✅ PELANGGAN TETAP BERLANGGANAN (LOYAL)")
    except Exception as e:
        st.error(f"Terjadi kesalahan kesesuaian kolom: {e}")
        st.info("Tips: Pastikan semua kolom inputan sudah lengkap sesuai struktur dataset.")