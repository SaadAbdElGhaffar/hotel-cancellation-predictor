import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np

# Local imports
from src.utils.helpers import (load_model, create_prediction_input, 
                              predict_cancellation)

# Load models
model_path = "models/hotel_reservations_model.pkl"
pipeline_path = "models/fitted_fullpipeline1.pkl"

model = load_model(model_path)
pipeline = load_model(pipeline_path)

# Create sample input
input_data = create_prediction_input(
    no_of_adults=2,
    no_of_children=1,
    no_of_weekend_nights=1,
    no_of_week_nights=3,
    type_of_meal_plan="Meal Plan 1",
    required_car_parking_space=0,
    room_type_reserved="Room_Type 1",
    lead_time=120,
    arrival_year=2018,
    arrival_month=8,
    arrival_date=15,
    market_segment_type="Online",
    repeated_guest=0,
    no_of_previous_cancellations=0,
    no_of_previous_bookings_not_canceled=2,
    avg_price_per_room=85.5,
    no_of_special_requests=1
)

# Make prediction
result = predict_cancellation(model, pipeline, input_data)

# Display results
print(f"Prediction: {result['prediction_label']}")
print(f"Probability of Cancellation: {result['probability_canceled']:.2%}")
print(f"Probability of No Cancellation: {result['probability_not_canceled']:.2%}")