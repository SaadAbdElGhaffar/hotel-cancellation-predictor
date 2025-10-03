import streamlit as st
import requests
import pandas as pd

# FastAPI endpoint URL (update if deployed elsewhere)
API_URL = "http://127.0.0.1:8000/predict"  # Assuming local FastAPI server

st.title("Hotel Reservations Prediction")

# Input form for features
no_of_adults = st.number_input("Number of Adults", min_value=0, value=2)
no_of_children = st.number_input("Number of Children", min_value=0, value=0)
no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, value=1)
no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, value=2)
type_of_meal_plan = st.selectbox("Type of Meal Plan", ["Meal Plan 1", "Not Selected", "Meal Plan 2", "Meal Plan 3"])
required_car_parking_space = st.selectbox("Required Car Parking Space", [0, 1])
room_type_reserved = st.selectbox("Room Type Reserved", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
lead_time = st.number_input("Lead Time (days)", min_value=0, value=224)
arrival_year = st.number_input("Arrival Year", min_value=2017, value=2017)
arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, value=10)
arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, value=2)
market_segment_type = st.selectbox("Market Segment Type", ["Offline", "Online", "Corporate", "Direct", "Complementary"])
repeated_guest = st.selectbox("Repeated Guest", [0, 1])
no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, value=0)
no_of_previous_bookings_not_canceled = st.number_input("Number of Previous Bookings Not Canceled", min_value=0, value=0)
avg_price_per_room = st.number_input("Average Price Per Room", min_value=0.0, value=65.0)
no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, value=0)

if st.button("Predict"):
    # Prepare input data
    input_data = {
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": required_car_parking_space,
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_date": arrival_date,
        "market_segment_type": market_segment_type,
        "repeated_guest": repeated_guest,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests
    }
    
    # Send request to FastAPI
    response = requests.post(API_URL, json=input_data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Booking Status: {result['prediction']}")
        st.write("Probabilities:")
        st.write(result['probability'])
    else:
        st.error(f"Error: {response.text}")

# To run: streamlit run app.py
# Make sure FastAPI is running first with: cd ../api && uvicorn app:app --reload