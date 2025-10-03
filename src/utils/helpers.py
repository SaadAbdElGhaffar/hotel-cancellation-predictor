"""
Helper utilities for hotel reservations prediction.
"""
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


def save_model(model, filepath):
    if filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    else:
        joblib.dump(model, filepath)


def load_model(filepath):
    if filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    else:
        model = joblib.load(filepath)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics_dict = {
        'accuracy': accuracy,
        'f1_score': f1,
    }
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        metrics_dict['roc_auc'] = roc_auc
    
    return metrics_dict


def create_prediction_input(no_of_adults=2, no_of_children=0, no_of_weekend_nights=1,
                          no_of_week_nights=2, type_of_meal_plan="Meal Plan 1",
                          required_car_parking_space=0, room_type_reserved="Room_Type 1",
                          lead_time=85, arrival_year=2018, arrival_month=10,
                          arrival_date=2, market_segment_type="Online",
                          repeated_guest=0, no_of_previous_cancellations=0,
                          no_of_previous_bookings_not_canceled=0,
                          avg_price_per_room=65.0, no_of_special_requests=0):
    """
    Create a properly formatted input for model prediction.
    
    Args:
        Various hotel reservation parameters
    
    Returns:
        pd.DataFrame: Single row DataFrame with all required features
    """
    input_data = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'market_segment_type': market_segment_type,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests
    }
    
    return pd.DataFrame([input_data])


def predict_cancellation(model, pipeline, input_data):
    # Preprocess the input data
    X_processed = pipeline.transform(input_data)
    
    # Make prediction
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0] if hasattr(model, 'predict_proba') else None
    
    result = {
        'prediction': int(prediction),
        'prediction_label': 'Canceled' if prediction == 1 else 'Not Canceled',
    }
    
    if probability is not None:
        result['probability_not_canceled'] = float(probability[0])
        result['probability_canceled'] = float(probability[1])
    
    return result


def get_data_summary(df):
    """
    Get a comprehensive summary of the dataset.
    
    Args:
        df: DataFrame to summarize
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {}
    }
    
    # Get categorical summary
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        summary['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'top_values': df[col].value_counts().head().to_dict()
        }
    
    return summary


def validate_input_data(input_data):
    """
    Validate input data for prediction.
    
    Args:
        input_data: DataFrame with input features
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    required_columns = [
        'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
        'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
        'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
        'arrival_date', 'market_segment_type', 'repeated_guest',
        'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
        'avg_price_per_room', 'no_of_special_requests'
    ]
    
    error_messages = []
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in input_data.columns]
    if missing_columns:
        error_messages.append(f"Missing required columns: {missing_columns}")
    
    # Check data types and ranges
    if 'no_of_adults' in input_data.columns:
        if input_data['no_of_adults'].iloc[0] < 0:
            error_messages.append("Number of adults must be non-negative")
    
    if 'no_of_children' in input_data.columns:
        if input_data['no_of_children'].iloc[0] < 0:
            error_messages.append("Number of children must be non-negative")
    
    if 'arrival_month' in input_data.columns:
        month = input_data['arrival_month'].iloc[0]
        if not (1 <= month <= 12):
            error_messages.append("Arrival month must be between 1 and 12")
    
    if 'arrival_date' in input_data.columns:
        date = input_data['arrival_date'].iloc[0]
        if not (1 <= date <= 31):
            error_messages.append("Arrival date must be between 1 and 31")
    
    if 'avg_price_per_room' in input_data.columns:
        if input_data['avg_price_per_room'].iloc[0] < 0:
            error_messages.append("Average price per room must be non-negative")
    
    is_valid = len(error_messages) == 0
    return is_valid, error_messages


def format_model_performance(metrics_dict):
    """
    Format model performance metrics for display.
    
    Args:
        metrics_dict: Dictionary with performance metrics
    
    Returns:
        str: Formatted performance string
    """
    performance_str = "Model Performance Summary:\n"
    performance_str += "=" * 30 + "\n"
    
    for metric, value in metrics_dict.items():
        if isinstance(value, float):
            performance_str += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"
        else:
            performance_str += f"{metric.replace('_', ' ').title()}: {value}\n"
    
    return performance_str