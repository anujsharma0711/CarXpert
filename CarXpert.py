import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from model.filter import get_matching_cars

# Load trained model
model = joblib.load("model/car_price_model.pkl")
dataset=pd.read_csv("data/cleaned_cars.csv")

st.set_page_config(page_title=" CarXpert - Car Resale Price Estimator", layout="centered", page_icon='🚘')

# --- Custom CSS ---
st.markdown("""<style>
    .block-container {
        padding-top: 2.3rem; /* Reduce from default ~6rem */
    }
            
    html, body {
        background-color: #3E4B3A !important;
        color: #FFFFFF !important;
        font-family: 'Times New Roman', serif !important;
    }
            
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stSidebar"] {
        background-color: #3E4B3A !important;
    }
            
    .main {
        background-color: #4A5D45 !important;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
            
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #FFFFFF !important;
        font-family: 'Times New Roman', serif !important;
    }
            
    .stButton>button {
        background-color: #FFFFFF !important;
        color: #3E4B3A !important;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        font-weight: bold;
    }
            
    .stButton>button:hover {
        background-color: #DDDDDD !important;
    }
            
    input, select, textarea {
        background-color: #E0E0D1 !important;
        color: #000000 !important;
    }
            
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div[role="button"],
    div[data-baseweb="popover"] li,
    div[data-baseweb="input"] input {
        background-color: #E0E0D1 !important;
        color: #000000 !important;
    }
            
    input::placeholder, textarea::placeholder,
    div[data-baseweb="input"] input::placeholder {
        color: #000000 !important;
        opacity: 1 !important;
    }
            
    [data-baseweb="radio"] input[type="radio"]:checked + div {
        color: #000000 !important;
    }
            
    [data-baseweb="radio"] label > div {
        color: #FFFFFF !important;
    }
            
    [data-testid="stSlider"] .st-bx,
    [data-testid="stSlider"] .st-c5 {
        background-color: #BFA181 !important;
    }
            
    div[data-baseweb="select"] svg {
        fill: #000000 !important;
    }     

    footer {visibility: hidden;}
    .stNumberInput { margin-bottom: 0.3rem !important; }
</style>""", unsafe_allow_html=True)

# --- Title ---
st.markdown("""
<h1 style='text-align: center; margin: 0; padding: 0;'>🚘 CarXpert 🚘</h1>
<p style='text-align: center; margin: 0; padding: 0; font-size: 16px; color: #ccc;'>&quot;Your Guide to Car’s True Worth&quot;</p>
""", unsafe_allow_html=True)
st.markdown("<h6></h6>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 30px;'>Car Resale Value Estimator with Smart Recommendations</h2>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 0; padding: 0;'>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 22px; text-align: center;'><b>Fill out the car details below to get estimated resale value and suggestions:</b></p>", unsafe_allow_html=True)

# --- Form ---
with st.form("car_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div style='font-size:20px;'>Fuel Type</div>", unsafe_allow_html=True)
        fuel_type = st.selectbox("", ["Petrol", "Diesel", "CNG", "LPG", "Hybrid"])

        st.markdown("<div style='font-size:20px;'>Transmission</div>", unsafe_allow_html=True)
        transmission = st.radio("", ["Manual", "Automatic"])

        st.markdown("<div style='font-size:20px;'>Owner Type</div>", unsafe_allow_html=True)
        owner_type = st.selectbox("", ["First", "Second", "Third"])

        st.markdown("<div style='font-size:20px;'>Seating Capacity</div>", unsafe_allow_html=True)
        seating_capacity = st.number_input("", min_value=2, max_value=8, step=1)

        st.markdown("<div style='font-size:20px;'>Car Age (in years)</div>", unsafe_allow_html=True)
        car_age = st.slider("", min_value=0, max_value=20, step=1)

    with col2:
        st.markdown("<div style='font-size:20px;'>Engine Capacity (in CC)</div>", unsafe_allow_html=True)
        engine_cc = st.number_input("", min_value=500, max_value=5000)

        st.markdown("<div style='font-size:20px;'>Fuel Tank Capacity (in liters)</div>", unsafe_allow_html=True)
        fuel_tank_cap = st.number_input("", min_value=10, max_value=200)

        st.markdown("<div style='font-size:20px;'>Max Power (in bhp)</div>", unsafe_allow_html=True)
        max_power = st.number_input("", min_value=20.0, max_value=700.0)

        st.markdown("<div style='font-size:20px;'>Kilometers Driven</div>", unsafe_allow_html=True)
        kilometer = st.number_input("", min_value=0, max_value=5000000, step=500)

        st.markdown("<div style='font-size:20px;'>Location (Optional)</div>", unsafe_allow_html=True)
        location=st.selectbox(
            "",["Pune", "Ludhiana", "Lucknow", "Mangalore", "Mumbai", "Coimbatore", "Bangalore", "Delhi", "Raipur",
                "Kanpur", "Patna", "Vadodara", "Hyderabad", "Yamunanagar", "Gurgaon", "Jaipur", "Deoghar", "Agra",
                "Goa", "Warangal", "Jalandhar", "Noida", "Ahmedabad", "Mohali", "Ghaziabad", "Kolkata", "Zirakpur",
                "Nagpur", "Thane", "Faridabad", "Ranchi", "Chandigarh", "Amritsar", "Chennai", "Navi Mumbai", "Udupi",
                "Jamshedpur", "Aurangabad", "Rudrapur", "Nashik", "Varanasi", "Salem", "Dehradun", "Valsad", "Haldwani",
                "Dharwad", "Surat", "Indore", "Karnal", "Panchkula", "Mysore", "Rohtak", "Ambala Cantt", "Samastipur",
                "Panvel", "Purnea", "Bhubaneswar", "Kheda", "Kollam", "Meerut", "Ernakulam", "Kharar", "Mirzapur",
                "Bhopal", "Gorakhpur", "Guwahati", "Allahabad", "Muzaffurpur", "Faizabad", "Kota", "Pimpri-Chinchwad",
                "Dak. Kannada", "Ranga Reddy", "Bulandshahar", "Roorkee"]
)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1.6, 1])
    with col_btn2:
        st.markdown("<h10></h10>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Estimate Price & Get Recommendations")

# --- After Submission ---
if submitted:
    with st.spinner("⏳ Estimating resale value and fetching recommendations..."):
        time.sleep(1.5)

        # --- Feature encoding to match model ---
        trans_encoded = 1 if transmission == "Manual" else 0
        owner_encoded = {"First": 0, "Second": 1, "Third": 2}[owner_type]

        # One-hot fuel encoding (match model columns)
        fuel_vector = [
            1 if fuel_type == "Diesel" else 0,
            1 if fuel_type == "Hybrid" else 0,
            1 if fuel_type == "LPG" else 0,
            1 if fuel_type == "Petrol" else 0,
            1 if fuel_type == "CNG" else 0  # model trained on Petrol+CNG
        ]

        # Final input list (same order as model expects)
        input_data = [[
            kilometer, *fuel_vector, trans_encoded, owner_encoded, seating_capacity,
            engine_cc, max_power,  car_age, fuel_tank_cap
        ]]

        # Predict log(price)
        log_pred_price = model.predict(input_data)[0]

        # Convert back to actual price
        predicted_price = np.expm1(log_pred_price)

        # --- Prediction ---
        # predicted_price = model.predict(input_data)[0]
        price_range = (predicted_price * 0.9, predicted_price * 1.1)

        user_input_dict = {
            'Location': location,
            'Fuel Type': fuel_type,
            'Transmission': transmission,
            'Owner': owner_type,
            'Seating Capacity': seating_capacity,
            'Car Age': car_age,
            'Engine CC': engine_cc,
            'Max Power': max_power,
            'Fuel Tank Capacity': fuel_tank_cap,
            'Kilometer': kilometer
        }

        # --- Display Results ---
        st.markdown("### ✅ Input Summary")
        st.success(f"""
        - **Fuel Type**: {fuel_type}  
        - **Transmission**: {transmission}  
        - **Owner Type**: {owner_type}  
        - **Seating Capacity**: {seating_capacity}  
        - **Car Age**: {car_age} years  
        - **Engine CC**: {engine_cc}  
        - **Fuel Tank Capacity**: {fuel_tank_cap}
        - **Max Power**: {max_power} bhp  
        - **Kilometers Driven**: {kilometer}  
        - **Location**: {location or 'Not specified'}  
        """)

        st.markdown("### 💰 Estimated Resale Price")
        st.success(f"**Estimated Resale Price**: ₹ {predicted_price:,.0f}")
        st.success(f"**Estimated Resale Price Range**: ₹ {price_range[0]:,.0f} – ₹ {price_range[1]:,.0f}")

        recommended_cars, message = get_matching_cars(user_input_dict, price_range, dataset, predicted_price)

        st.markdown(f"### Suggested Cars around ₹{price_range[0]:,.0f} – ₹ {price_range[1]:,.0f}")
        st.info(f"{message}")

        if recommended_cars.empty:
            st.warning("Sorry! Couldn't find any suggestions.")
            st.info("Try modifying your input preferences for better matches.")
        else:
            for idx, row in recommended_cars.iterrows():
                car_name = f"{row['Company']} {row['Base Model']}"

                # Decode Transmission
                transmission = "Automatic" if row['Transmission'] == 1 else "Manual"

                # Decode Owner
                owner_mapping = {
                    0: "First Owner",
                    1: "Second Owner",
                    2: "Third Owner",
                    3: "Fourth & Above Owner"
                }
                owner_status = owner_mapping.get(row['Owner'], "Unknown")

                with st.expander(f"🚗 {car_name}"):
                    fuel_type = "Unknown"
                    for ft in ["Petrol", "Diesel", "CNG", "LPG", "Hybrid", "Petrol+CNG"]:
                        col_name = f"Fuel_{ft}"
                        if col_name in row and row[col_name] == 1:
                            fuel_type = ft
                            break
                    st.markdown(f"""
                    - **Location**: {row['Location']}
                    - **Fuel Type**: {fuel_type}
                    - **Transmission**: {transmission}
                    - **Owner**: {owner_status}
                    - **Car Age**: {row['Car Age']} years
                    - **Seating Capacity**: {row['Seating Capacity']}
                    - **Kilometers Driven**: {int(row['Kilometer']):,} km
                    - **Max Power**: {row['Max Power']} bhp
                    - **Fuel Tank Capacity**: {row['Fuel Tank Capacity']} L
                    - **Estimated Resale Price**: ₹{int(row['Price']):,}
                    """)
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #E0E0D1; font-size: 14px;'>"
    "© 2025 <strong>CarXpert</strong> by Anuj Sharma. All rights reserved."
    "</div>",
    unsafe_allow_html=True
)