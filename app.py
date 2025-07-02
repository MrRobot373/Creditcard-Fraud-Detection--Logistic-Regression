import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

# Load model and encoders
model = joblib.load("Creditcard_fraud_detection_model.jb")
encoder = joblib.load("lable_encoder.jb")

# Haversine function for distance
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# ------------------- Page Setup -------------------
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.markdown("<h1 style='color:#3498db;'>🔐 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("Fill in the transaction details below:")

# ------------------- Input Form -------------------
with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    with col1:
        merchant = st.text_input("Merchant Name", placeholder="e.g. fraud_Jast-McDermott")
        category = st.text_input("Category", placeholder="e.g. shopping_pos")
        amt = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cc_num = st.text_input("Credit Card Number")

    with col2:
        lat = st.number_input("Latitude", format="%.6f")
        long = st.number_input("Longitude", format="%.6f")
        merch_lat = st.number_input("Merchant Latitude", format="%.6f")
        merch_long = st.number_input("Merchant Longitude", format="%.6f")
        hour = st.slider("Transaction Hour", 0, 23, 12)
        day = st.slider("Transaction Day", 1, 31, 15)
        month = st.slider("Transaction Month", 1, 12, 6)

    submit = st.form_submit_button("🚨 Check For Fraud")

# ------------------- Prediction Logic -------------------
if submit:
    if merchant and category and cc_num:
        distance = haversine(lat, long, merch_lat, merch_long)

        input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                  columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])

        # Encode categorical values
        for col in ['merchant', 'category', 'gender']:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1  # Unknown categories fallback

        # Hash credit card number
        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "🚩 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"

        # Show map
        st.subheader("📍 Transaction Map")
        mid_lat = (lat + merch_lat) / 2
        mid_long = (long + merch_long) / 2
        m = folium.Map(location=[mid_lat, mid_long], zoom_start=12)

        folium.Marker([lat, long], tooltip="User Location", icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker([merch_lat, merch_long], tooltip="Merchant Location", icon=folium.Icon(color="green")).add_to(m)
        folium.PolyLine(locations=[[lat, long], [merch_lat, merch_long]], color='red').add_to(m)
        st_folium(m, width=700, height=450)

        # Show result
        st.markdown("---")
        st.markdown(f"<h2 style='color: {'red' if prediction == 1 else 'green'}'>{result}</h2>", unsafe_allow_html=True)

    else:
        st.error("⚠️ Please fill all required fields.")
