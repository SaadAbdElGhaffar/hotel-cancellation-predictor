from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import sys
import os
import joblib

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all transformer classes for pickle compatibility
from src.data import (
    FullPipeline1,
    ColumnSelector,
    QuarterTransformer,
    Transformation,
    OneHotEncodeColumns,
    LabelEncodeColumns,
    ScalingTransform,
    DropColumnsTransformer
)

# Fix for joblib loading issue: make FullPipeline1 available in __main__ namespace
import __main__
__main__.FullPipeline1 = FullPipeline1
__main__.ColumnSelector = ColumnSelector
__main__.QuarterTransformer = QuarterTransformer
__main__.Transformation = Transformation
__main__.OneHotEncodeColumns = OneHotEncodeColumns
__main__.LabelEncodeColumns = LabelEncodeColumns
__main__.ScalingTransform = ScalingTransform
__main__.DropColumnsTransformer = DropColumnsTransformer

app = FastAPI(title="Hotel Reservations Prediction API")

# Get the absolute path to the models directory
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Load the pre-fitted pipeline and model
full_pipeline = joblib.load(os.path.join(models_dir, "fitted_fullpipeline1.joblib"))

with open(os.path.join(models_dir, 'hotel_reservations_model.pkl'), 'rb') as f:
    model = pickle.load(f)

def transform_features_only(df):
    """
    Helper function to transform only features (X) without target variable (y).
    This is needed for prediction since FullPipeline1.transform() expects both X and y.
    """
    return full_pipeline.full_pipeline.transform(df)

# Define the input data model using Pydantic
class BookingInput(BaseModel):
    no_of_adults: int
    no_of_children: int
    no_of_weekend_nights: int
    no_of_week_nights: int
    type_of_meal_plan: str
    required_car_parking_space: int
    room_type_reserved: str
    lead_time: int
    arrival_year: int
    arrival_month: int
    arrival_date: int
    market_segment_type: str
    repeated_guest: int
    no_of_previous_cancellations: int
    no_of_previous_bookings_not_canceled: int
    avg_price_per_room: float
    no_of_special_requests: int

@app.post("/predict")
async def predict_booking(input_data: BookingInput):
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    
    # Transform features using our helper function
    processed_data = transform_features_only(df)
    
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)[0]
    
    status = full_pipeline.inverse_y(prediction)[0]
    label_0 = full_pipeline.inverse_y([0])[0]
    label_1 = full_pipeline.inverse_y([1])[0]
    
    return {
        "prediction": status,
        "probability": {
            label_0: float(prediction_proba[0]),
            label_1: float(prediction_proba[1])
        }
    }

@app.get("/")
async def root():
    return {"message": "Welcome to Hotel Reservations Prediction API. Use /predict endpoint for predictions."}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)