# 💳 Fraud Detection System

A web-based application built using **Streamlit** and **LightGBM** to predict fraudulent credit card transactions based on geolocation, time, and transaction metadata.

---

## 🚀 Features

- 🧠 Predicts fraudulent vs. legitimate transactions  
- 🗺️ Calculates geodesic distance using coordinates  
- 💻 Streamlit interface for interactive inputs  
- 📦 Uses pre-trained LightGBM model and label encoders  
- 🔐 Hashing applied on credit card numbers for privacy  

---

## 📁 Project Structure
.
├── app.py # Streamlit web app
├── Untitled0.ipynb # Data processing and model training
├── fraud_detection_model.jb # Trained model
├── label_encoder.jb # Label encoders 


---

## 🧾 Input Fields in UI

- **Merchant Name**
- **Transaction Category**
- **Transaction Amount**
- **User Latitude & Longitude**
- **Merchant Latitude & Longitude**
- **Transaction Hour, Day, Month**
- **Gender**
- **Credit Card Number (hashed internally)**

