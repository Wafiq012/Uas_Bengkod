# 1. Masukkan input dari user
user_input = {
    'gender': gender, 'age': age, 'country': country, 'city': city,
    'is_premium_user': is_premium_user, 'total_visits': total_visits,
    'total_spent': total_spent, 'satisfaction_score': satisfaction_score
}

# 2. Daftar semua kolom yang diminta oleh model Anda (Sudah dipastikan lengkap)
missing_cols = [
    'acquisition_channel', 'customer_id', 'email_click_rate', 'support_tickets',
    'discount_used', 'last_3_month_purchase_freq', 'device_type', 'payment_method',
    'subscription_type', 'avg_session_time', 'pages_per_session', 'nps_score',
    'delivery_delay_days', 'lifetime_value', 'email_open_rate', 'marketing_spent_per_user',
    'refund_requested', 'avg_order_value', 'marketing_spent_per_user'
]

# 3. Gabungkan input user dengan nilai default agar model menerima struktur yang utuh
for col in missing_cols:
    if col not in user_input:
        if 'score' in col or 'rate' in col or 'time' in col or 'value' in col or 'spent' in col or 'freq' in col or 'order' in col:
            user_input[col] = 0.0  # Angka desimal
        elif 'tickets' in col or 'days' in col or 'used' in col or 'visits' in col:
            user_input[col] = 0    # Angka bulat
        else:
            user_input[col] = ""   # Teks kosong

# Jadikan ke bentuk DataFrame
input_df = pd.DataFrame([user_input])

# --- MEMASTIKAN NAMA KOLOM UNTUK TOTAL VISITS & SATISFACTION SCORE SESUAI DUPLIKASI MODEL ---
# Jika model Anda dulu dilatih menggunakan spasi (bukan underscore), baris ini akan mengamankannya:
if 'total visits' not in input_df.columns and 'total_visits' in input_df.columns:
    input_df['total visits'] = input_df['total_visits']
if 'satisfaction score' not in input_df.columns and 'satisfaction_score' in input_df.columns:
    input_df['satisfaction score'] = input_df['satisfaction_score']

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
