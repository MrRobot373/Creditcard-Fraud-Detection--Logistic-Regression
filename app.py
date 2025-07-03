import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

model = joblib.load("Creditcard_fraud_detection_model.jb")
encoder = joblib.load("lable_encoder.jb")

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.markdown("<h1 style='color:#3498db;'>🔐 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)

left, right = st.columns([1, 1])

with left.form("transaction_form"):
    st.subheader("📝 Transaction Details")

    merchant = st.text_input("Merchant Name", placeholder="e.g. fraud_Jast-McDermott")
    category = st.text_input("Category", placeholder="e.g. shopping_pos")
    amt = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    gender = st.selectbox("Gender", ["Male", "Female"])
    cc_num = st.text_input("Credit Card Number")

    lat = st.number_input("Latitude", format="%.6f")
    long = st.number_input("Longitude", format="%.6f")
    merch_lat = st.number_input("Merchant Latitude", format="%.6f")
    merch_long = st.number_input("Merchant Longitude", format="%.6f")
    hour = st.slider("Transaction Hour", 0, 23, 12)
    day = st.slider("Transaction Day", 1, 31, 15)
    month = st.slider("Transaction Month", 1, 12, 6)

    submitted = st.form_submit_button("🚨 Check For Fraud")

if submitted:
    if merchant and category and cc_num:
        distance = haversine(lat, long, merch_lat, merch_long)

        input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                  columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])

        for col in ['merchant', 'category', 'gender']:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        prediction = model.predict(input_data)[0]
        st.session_state["fraud_result"] = prediction
        st.session_state["show_map"] = True
        st.session_state["locations"] = {"user": (lat, long), "merchant": (merch_lat, merch_long)}
    else:
        st.error("⚠️ Please fill all required fields.")
        st.session_state["fraud_result"] = None
        st.session_state["show_map"] = False

with right:
    st.subheader("🔎 Prediction Result")

    if "fraud_result" in st.session_state and st.session_state["fraud_result"] is not None:
        prediction = st.session_state["fraud_result"]
        result = "🚩 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        color = "red" if prediction == 1 else "green"

        st.markdown(f"<h2 style='color:{color}'>{result}</h2>", unsafe_allow_html=True)
    else:
        st.info("Submit a transaction to see the result.")

if st.session_state.get("show_map", False):
    st.subheader("📍 Transaction Map")
    user_loc = st.session_state["locations"]["user"]
    merchant_loc = st.session_state["locations"]["merchant"]
    mid_lat = (user_loc[0] + merchant_loc[0]) / 2
    mid_long = (user_loc[1] + merchant_loc[1]) / 2

    m = folium.Map(location=[mid_lat, mid_long], zoom_start=12)
    folium.Marker(user_loc, tooltip="User Location", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker(merchant_loc, tooltip="Merchant Location", icon=folium.Icon(color="green")).add_to(m)
    folium.PolyLine(locations=[user_loc, merchant_loc], color='red').add_to(m)

    st_folium(m, width=700, height=450)
