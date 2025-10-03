"""
Data loading utilities for hotel reservations dataset.
"""
import pandas as pd
import numpy as np
import os


def load_hotel_data(data_path):
    df = pd.read_csv(data_path)
    return df


def get_feature_columns(df):
    num_cols = df.select_dtypes(include=['number'])\
                .columns.difference(['Booking_ID']).tolist()
    cat_cols = df.select_dtypes(include=['object', 'category'])\
                .columns.difference(['Booking_ID']).tolist()
    
    return num_cols, cat_cols


def show_unique_values(df):
    for column in df.columns:
        unique_values = df[column].unique()
        print(f"Unique values in '{column}':\n{unique_values}\n")


def unique_counts(df):
    num_unique = df.nunique().sort_values(ascending=False)
    pct_unique = ((df.nunique().sort_values(ascending=False) / len(df)) * 100).round(3)
    pct_unique = pct_unique.astype(str) + '%'
    
    unique = pd.DataFrame({
        'Unique Count': num_unique,
        'Percentage Unique': pct_unique
    })

    return unique


def prepare_data_for_modeling(df):
    # Drop Booking_ID as it's just an identifier
    df_processed = df.drop('Booking_ID', axis=1, errors='ignore')
    
    # Create quarter feature from arrival_month
    conditions = [
        (df_processed['arrival_month'] <= 3),
        (df_processed['arrival_month'] > 3) & (df_processed['arrival_month'] <= 6),
        (df_processed['arrival_month'] > 6) & (df_processed['arrival_month'] <= 9),
        (df_processed['arrival_month'] >= 10)
    ]
    values = ['Q2', 'Q3', 'Q4', 'Q1']
    df_processed['quarter'] = np.select(conditions, values)
    
    return df_processed