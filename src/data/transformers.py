import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import yeojohnson
from sklearn.preprocessing import OrdinalEncoder

num_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
           'arrival_year','arrival_month','arrival_date','lead_time','no_of_previous_cancellations', 
          'no_of_previous_bookings_not_canceled','avg_price_per_room','no_of_special_requests']

class ColumnSelector(TransformerMixin, BaseEstimator):

    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X = X[self.columns]
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)


class Transformation(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, skew_threshold=1):
        
        self.cols = cols
        self.skew_threshold = skew_threshold
        self.skewed_columns = [] 

    def fit(self, X, y=None):
        
        if self.cols is None:
            self.cols = X.select_dtypes(include=[np.number]).columns
        
        self.skewed_columns = [
            col for col in self.cols if abs(X[col].skew()) > self.skew_threshold
        ]
        return self

    def transform(self, X):
        
        X_copy = X.copy()
        for col in self.skewed_columns:
            # Apply log transformation
            X_copy[col] = np.log1p(X_copy[col])
        return X_copy

    def fit_transform(self, X, y=None):
       
        return self.fit(X, y).transform(X)

class ScalingTransform(BaseEstimator, TransformerMixin):

    def __init__(self, cols, scaling_method):
        self.cols = cols
        self.scaler_ = None
        self.scaling_method = scaling_method

    def fit(self, X, y=None):
        if self.scaling_method == "std_scale":
            self.scaler_ = StandardScaler().fit(X.loc[:, self.cols])
        elif self.scaling_method == "min_max_scale":
            self.scaler_ = MinMaxScaler().fit(X.loc[:, self.cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.cols] = self.scaler_.transform(X_copy.loc[:, self.cols])
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)



class OneHotEncodeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        
        self.cols = cols
        self.encoder = None
        self.column_names = None

    def fit(self, X, y=None):
        
        self.encoder = OneHotEncoder(sparse=False, drop='first')
        self.encoder.fit(X[self.cols])
        self.column_names = self.encoder.get_feature_names_out(self.cols)
        return self

    def transform(self, X):
        
        X_copy = X.copy()

       
        encoded_data = self.encoder.transform(X_copy[self.cols])
        encoded_df = pd.DataFrame(encoded_data, columns=self.column_names, index=X_copy.index)

        
        X_copy = X_copy.drop(columns=self.cols)
        X_copy = pd.concat([X_copy, encoded_df], axis=1)

        return X_copy

    def fit_transform(self, X, y=None):
        
        self.fit(X, y)
        return self.transform(X)


class LabelEncodeColumns(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols
        self.encoders_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self.encoders_[col] = encoder
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders_.items():
            X_copy[col] = encoder.transform(X_copy[col])
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders_.items():
            X_copy[col] = encoder.inverse_transform(X_copy[col])
        return X_copy


class OrdinalEncodeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols, categories=None):

        self.cols = cols
        self.categories = categories
        self.encoder = None

    def fit(self, X, y=None):

        self.encoder = OrdinalEncoder(categories=self.categories)
        self.encoder.fit(X[self.cols])
        return self

    def transform(self, X):

        X_copy = X.copy()


        encoded_data = self.encoder.transform(X_copy[self.cols])


        encoded_df = pd.DataFrame(encoded_data, columns=self.cols, index=X_copy.index)


        X_copy = X_copy.drop(columns=self.cols)
        X_copy = pd.concat([X_copy, encoded_df], axis=1)

        return X_copy

    def fit_transform(self, X, y=None):

        self.fit(X, y)
        return self.transform(X)


class DropColumnsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None):
        self.cols = cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.cols is None:
            return X
        else:
            return X.drop(self.cols,axis=1)

class QuarterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='arrival_month'):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        conditions = [
            (X[self.column] <= 3),
            (X[self.column] > 3) & (X[self.column] <= 6),
            (X[self.column] > 6) & (X[self.column] <= 9),
            (X[self.column] >= 10)
        ]
        values = ['Q2', 'Q3', 'Q4', 'Q1']
        X['quarter'] = np.select(conditions, values)
        return X

class FullPipeline1:
    def __init__(self):
        self.all_cols = num_cols + ['type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
            'market_segment_type', 'repeated_guest']

        self.drop_cols = ['arrival_year','arrival_date','room_type_reserved_Room_Type 2','room_type_reserved_Room_Type 3','type_of_meal_plan_Not Selected','type_of_meal_plan_Meal Plan 3',
                         'arrival_month','room_type_reserved_Room_Type 5']

        self.one_hot_encode_cols = ['market_segment_type','type_of_meal_plan', 'room_type_reserved','quarter']

        self.label_encode = ['booking_status']
        self.scale_cols = num_cols

        self.full_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=self.all_cols)),

            ('quarter_transform', QuarterTransformer(column='arrival_month')),
            ('power_transformation', Transformation(cols=num_cols)),
            ('one_hot_encode', OneHotEncodeColumns(cols=self.one_hot_encode_cols)),

            ('scaling', ScalingTransform(cols=self.scale_cols,
                                         scaling_method="min_max_scale")),
            ('drop_cols', DropColumnsTransformer(cols=self.drop_cols))
        ])

        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=['booking_status'])),

            ('label_encode', LabelEncodeColumns(cols=self.label_encode))
        ])

    def fit_transform(self, X_train, y_train):
        X_train = self.full_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train

    def transform(self, X_test, y_test):
        X_test = self.full_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test
        
    def inverse_y(self, y_pred):
        return self.y_pipeline.named_steps['label_encode'].inverse_transform(y_pred)
    

