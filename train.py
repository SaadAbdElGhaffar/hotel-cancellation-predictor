"""
Main training script for hotel reservations prediction model.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Local imports
from src.data.data_loader import load_hotel_data, get_feature_columns, prepare_data_for_modeling
from src.data.transformers import FullPipeline
from src.models.model_training import (get_base_models, baseline_classifier,
                                     evaluate_models, train_final_model)
from src.visualization.plots import (plot_target_distribution, plot_model_comparison,
                                   plot_evaluation_metrics)
from src.utils.helpers import save_model, evaluate_model


def main():
    # Load data
    data_path = "data/Hotel Reservations.csv"
    df = load_hotel_data(data_path)
    
    # Get feature columns
    num_cols, cat_cols = get_feature_columns(df)
    
    # Prepare data
    df_processed = prepare_data_for_modeling(df)
    
    # Split data
    X = df_processed.drop('booking_status', axis=1)
    y = df_processed['booking_status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing
    pipeline = FullPipeline(num_cols=num_cols)
    
    X_train_processed = pipeline.fit_transform(X_train)
    y_train_processed = pipeline.transform_target(y_train)
    
    X_test_processed = pipeline.transform(X_test)
    y_test_processed = pipeline.transform_target(y_test)
    
    # Baseline model
    baseline_results = baseline_classifier(
        X_train_processed, y_train_processed, 
        X_test_processed, y_test_processed
    )
    
    # Model comparison
    models = get_base_models()
    
    results_df = evaluate_models(
        models, X_train_processed, y_train_processed,
        X_test_processed, y_test_processed
    )
    
    # Train final model
    final_model = train_final_model(
        X_train_processed, y_train_processed,
        model_type='stacking',
        use_smote=True
    )
    
    # Final evaluation
    final_metrics = evaluate_model(
        final_model, X_test_processed, y_test_processed,
        model_name="Final Stacking Classifier"
    )
    
    # Save models
    os.makedirs("models", exist_ok=True)
    
    save_model(final_model, "models/hotel_reservations_model.pkl")
    save_model(pipeline, "models/fitted_fullpipeline1.pkl")
    
    return final_model, pipeline, final_metrics


if __name__ == "__main__":
    model, pipeline, metrics = main()