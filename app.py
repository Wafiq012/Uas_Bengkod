import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("🎯 Aplikasi Prediksi Churn Pelanggan")
st.write("UAS Bengkel Koding Data Science")

# 1. Memuat model dengan pengaman
model = None
try:
    model = joblib.load('best_churn_model_pipeline.pkl')
except Exception as e:
    st.error(f"Gagal memuat file model pkl: {e}")

# 2. Membuat Form Input di Sidebar
st.sidebar.header("📋 Input Fitur Karakteristik Pelanggan")

gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.sidebar.number_input("Usia", min_value=18, max_value=100, value=30)
country = st.sidebar.text_input("Negara Asal", "United States")
city = st.sidebar.text_input("Kota", "New York")
is_premium_user = st.sidebar.selectbox("Premium User?", [0, 1])
total_visits = st.sidebar.number_input("Total Kunjungan", min_value=0, value=10)
total_spent = st.sidebar.number_input("Total Pengeluaran", min_value=0.0, value=50000.0)
satisfaction_score = st.sidebar.slider("Skor Kepuasan", 1.0, 5.0, 3.5)

# 3. Definisikan susunan SEMUA KOLOM secara urut (Sesuai urutan latih model Anda)
feature_columns = [
    'customer_id', 'age', 'gender', 'country', 'city',
    'subscription_type', 'discount_used', 'total_visits', 'total_spent',
    'payment_method', 'avg_session_time', 'pages_per_session',
    'last_3_month_purchase_freq', 'avg_order_value', 'is_premium_user',
    'marketing_spent_per_user', 'satisfaction_score', 'nps_score',
    'email_open_rate', 'email_click_rate', 'support_tickets',
    'delivery_delay_days', 'refund_requested', 'acquisition_channel', 'lifetime_value'
]

# 4. Buat dictionary kosong, lalu isi berdasarkan input user atau nilai default
user_input = {}
for col in feature_columns:
    # Ambil dari input manual user jika ada
    if col == 'gender': user_input[col] = gender
    elif col == 'age': user_input[col] = age
    elif col == 'country': user_input[col] = country
    elif col == 'city': user_input[col] = city
    elif col == 'is_premium_user': user_input[col] = is_premium_user
    elif col == 'total_visits': user_input[col] = total_visits
    elif col == 'total_spent': user_input[col] = total_spent
    elif col == 'satisfaction_score': user_input[col] = satisfaction_score
    # Isi dengan nilai default yang aman jika tidak ada di form input
    else:
        if 'score' in col or 'rate' in col or 'time' in col or 'value' in col or 'spent' in col or 'freq' in col or 'order' in col:
            user_input[col] = 0.0
        elif 'tickets' in col or 'days' in col or 'used' in col:
            user_input[col] = 0
        else:
            user_input[col] = "Unknown"

# Ubah ke DataFrame dan PAKSA urutannya sesuai list 'feature_columns'
input_df = pd.DataFrame([user_input])
input_df = input_df[feature_columns]

# Duplikasi kolom jika model Anda menggunakan spasi (untuk berjaga-jaga)
input_df['total visits'] = input_df['total_visits']
input_df['satisfaction score'] = input_df['satisfaction_score']

st.subheader("Data Input Pelanggan:")
st.dataframe(input_df)

# 5. Tombol Eksekusi Prediksi
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
