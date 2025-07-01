# Creditcard-Fraud-Detection--Logistic-Regression
💳 Fraud Detection System
A web-based application built using Streamlit and LightGBM to predict fraudulent credit card transactions based on geolocation, time, and transaction metadata.

🚀 Features
User-friendly interface for fraud prediction

Uses a trained LightGBM model for accurate classification

Computes geodesic distance between transaction location and merchant location

Encodes categorical features with pre-trained label encoders

Simple deployment with Streamlit

📁 Project Structure
graphql
Copy
Edit
.
├── app.py                # Streamlit web app
├── Untitled0.ipynb       # Model training and preprocessing
├── fraud_detection_model.jb   # Trained LightGBM model (not included here)
├── label_encoder.jb           # Encoded label dictionary (not included here)
🧠 Model Input Features
The following inputs are required for fraud prediction:

merchant: Merchant name

category: Transaction category

amt: Transaction amount

lat, long: Transaction latitude and longitude

merch_lat, merch_long: Merchant's latitude and longitude

hour, day, month: Timestamp details of transaction

gender: Customer gender

cc_num: Credit card number (hashed internally)

🧮 Distance Calculation
The distance between the transaction and the merchant location is computed using the Haversine formula via geopy.

📦 Requirements
Install dependencies using pip:

bash
Copy
Edit
pip install -r requirements.txt
<details> <summary>Dependencies (as per notebook)</summary>
pandas

numpy

lightgbm

seaborn

matplotlib

scikit-learn

imbalanced-learn

joblib

streamlit

geopy

</details>
🛠️ How to Run
Make sure fraud_detection_model.jb and label_encoder.jb are present in the same directory.

Run the app:

bash
Copy
Edit
streamlit run app.py
Enter transaction details and click "Check For Fraud".
