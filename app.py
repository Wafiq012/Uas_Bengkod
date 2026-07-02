import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("🎯 Aplikasi Prediksi Churn Pelanggan")
st.write("UAS Bengkel Koding Data Science")

# Memuat model dengan pengaman try-except
model = None
try:
    model = joblib.load('best_churn_model_pipeline.pkl')
except Exception as e:
    st.error(f"Gagal memuat file model pkl: {e}")

st.sidebar.header("📋 Input Fitur Karakteristik Pelanggan")

gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.sidebar.number_input("Usia", min_value=18, max_value=100, value=30)
country = st.sidebar.text_input("Negara Asal", "United States")
city = st.sidebar.text_input("Kota", "New York")
is_premium_user = st.sidebar.selectbox("Premium User?", [0, 1])
total_visits = st.sidebar.number_input("Total Kunjungan", min_value=0, value=10)
total_spent = st.sidebar.number_input("Total Pengeluaran", min_value=0.0, value=50000.0)
satisfaction_score = st.sidebar.slider("Skor Kepuasan", 1.0, 5.0, 3.5)

# 1. Masukkan input dari user
user_input = {
    'gender': gender, 'age': age, 'country': country, 'city': city,
    'is_premium_user': is_premium_user, 'total_visits': total_visits,
    'total_spent': total_spent, 'satisfaction_score': satisfaction_score
}

# 2. Daftar semua kolom yang diminta oleh model Anda (berdasarkan error log)
missing_cols = [
    'acquisition_channel', 'customer_id', 'email_click_rate', 'support_tickets',
    'discount_used', 'last_3_month_purchase_freq', 'device_type', 'payment_method',
    'subscription_type', 'avg_session_time', 'pages_per_session', 'nps_score',
    'delivery_delay_days', 'lifetime_value', 'email_open_rate', 'marketing_spent_per_user',
    'refund_requested', 'avg_order_value'
]

# 3. Gabungkan input user dengan nilai default (0 atau string kosong) untuk kolom yang kurang
for col in missing_cols:
    if col not in user_input:
        if 'score' in col or 'rate' in col or 'time' in col or 'value' in col or 'spent' in col or 'freq' in col:
            user_input[col] = 0.0  # Isi dengan angka desimal jika berupa nilai/skor/rate
        elif 'tickets' in col or 'days' in col or 'used' in col:
            user_input[col] = 0    # Isi dengan angka bulat
        else:
            user_input[col] = ""   # Isi dengan teks kosong untuk kategori seperti device/payment

# Jadikan ke bentuk DataFrame
input_df = pd.DataFrame([user_input])

st.subheader("Data Input Pelanggan:")
st.dataframe(input_df)

if st.button("🚀 Prediksi Potensi Churn"):
    if model is None:
        st.error("Model tidak tersedia atau gagal dimuat.")
    else:
        try:
            prediction = model.predict(input_df)
            if prediction[0] == 1:
                st.error("⚠️ PELANGGAN BERPOTENSI CHURN (BERHENTI BERLANGGANAN)")
            else:
                st.success("✅ PELANGGAN TETAP BERLANGGANAN (LOYAL)")
        except Exception as e:
            st.error(f"Terjadi kesalahan kesesuaian kolom: {e}")
