"""
Model training utilities for hotel reservations prediction.
"""
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import (train_test_split, cross_validate, 
                                   GridSearchCV, StratifiedKFold)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                            GradientBoostingClassifier, AdaBoostClassifier,
                            BaggingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def get_base_models():
    """
    Get a list of base machine learning models for evaluation.
    
    Returns:
        list: List of initialized models
    """
    models = [
        LogisticRegression(random_state=83),
        DecisionTreeClassifier(random_state=83),
        RandomForestClassifier(random_state=83),
        ExtraTreesClassifier(random_state=83),
        GradientBoostingClassifier(random_state=83),
        XGBClassifier(random_state=83, use_label_encoder=False, eval_metric='logloss'),
        AdaBoostClassifier(random_state=83),
        BaggingClassifier(random_state=83),
        KNeighborsClassifier(),
        GaussianNB(),
    ]
    return models


def baseline_classifier(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a baseline dummy classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    
    Returns:
        dict: Baseline metrics
    """
    base = DummyClassifier()
    base.fit(X_train, y_train)
    
    y_pred_base = base.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_base)
    f1 = f1_score(y_test, y_pred_base)
    roc_auc = roc_auc_score(y_test, y_pred_base)
    
    print(f"Baseline model achieves accuracy = {accuracy:.4f}")
    print(f"Baseline model achieves F1 score = {f1:.4f}")
    print(f"Baseline model achieves ROC AUC = {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc
    }


def evaluate_models(models, X_train, y_train, X_test, y_test, cv=5):
    """
    Evaluate multiple models using cross-validation and test set.
    
    Args:
        models: List of models to evaluate
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        cv: Number of cross-validation folds
    
    Returns:
        pd.DataFrame: Results dataframe with model performance
    """
    results = []

    for model in models:
        start = time.time()
        
        # Cross-validation on training set
        cv_results = cross_validate(model, X_train, y_train, cv=cv,
                                  scoring=['accuracy', 'roc_auc', 'f1'], 
                                  return_train_score=True)
        
        train_mean_accuracy = np.mean(cv_results['train_accuracy'])
        train_mean_roc_auc = np.mean(cv_results['train_roc_auc'])
        train_mean_f1 = np.mean(cv_results['train_f1'])
        
        # Fit on full training set and evaluate on test set
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        
        test_accuracy = accuracy_score(y_test, test_preds)
        test_roc_auc = roc_auc_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds)
        
        # Store results
        results_dict = {
            'model': model.__class__.__name__,
            'train_accuracy': train_mean_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_mean_f1,
            'test_f1': test_f1,
            'train_roc_auc': train_mean_roc_auc,
            'test_roc_auc': test_roc_auc,
            'time': time.time() - start
        }
        results.append(results_dict)

    results_df = pd.DataFrame(results)
    results_df.set_index('model', inplace=True)
    results_df = results_df.sort_values(by='test_accuracy', ascending=False)

    return results_df


def tune_random_forest(X_train, y_train, cv=5):
    """
    Hyperparameter tuning for Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
    
    Returns:
        RandomForestClassifier: Best tuned model
    """
    param_grid_rf = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    
    rf_model = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, 
                                cv=cv, n_jobs=-1, verbose=1, scoring='accuracy')
    
    start_time = time.time()
    grid_search_rf.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Random Forest tuning completed in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {grid_search_rf.best_params_}")
    print(f"Best cross-validation score: {grid_search_rf.best_score_:.4f}")
    
    return grid_search_rf.best_estimator_


def tune_xgboost(X_train, y_train, cv=5):
    """
    Hyperparameter tuning for XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
    
    Returns:
        XGBClassifier: Best tuned model
    """
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb,
                                 cv=cv, n_jobs=-1, verbose=1, scoring='accuracy')
    
    start_time = time.time()
    grid_search_xgb.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"XGBoost tuning completed in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {grid_search_xgb.best_params_}")
    print(f"Best cross-validation score: {grid_search_xgb.best_score_:.4f}")
    
    return grid_search_xgb.best_estimator_


def create_stacking_classifier(best_rf, best_xgb):
    """
    Create a stacking classifier with the best models.
    
    Args:
        best_rf: Best Random Forest model
        best_xgb: Best XGBoost model
    
    Returns:
        StackingClassifier: Stacking ensemble model
    """
    base_models = [
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('et', ExtraTreesClassifier(random_state=42))
    ]
    
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(random_state=42),
        cv=5,
        stack_method='predict_proba'
    )
    
    return stacking_clf


def apply_smote_oversampling(X, y, random_state=42):
    """
    Apply SMOTE oversampling to balance the dataset.
    
    Args:
        X: Features
        y: Target
        random_state: Random state for reproducibility
    
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    oversample = SMOTE(random_state=random_state)
    X_resampled, y_resampled = oversample.fit_resample(X, y)
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled


def train_final_model(X_train, y_train, model_type='stacking', use_smote=True):
    """
    Train the final model with optional SMOTE oversampling.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train ('rf', 'xgb', 'stacking')
        use_smote: Whether to apply SMOTE oversampling
    
    Returns:
        object: Trained model
    """
    if use_smote:
        X_train, y_train = apply_smote_oversampling(X_train, y_train)
    
    if model_type == 'rf':
        model = tune_random_forest(X_train, y_train)
    elif model_type == 'xgb':
        model = tune_xgboost(X_train, y_train)
    elif model_type == 'stacking':
        # Train individual models first
        best_rf = tune_random_forest(X_train, y_train)
        best_xgb = tune_xgboost(X_train, y_train)
        
        # Create stacking classifier
        model = create_stacking_classifier(best_rf, best_xgb)
        
        print("Training stacking classifier...")
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Stacking classifier training completed in {end_time - start_time:.2f} seconds")
    else:
        raise ValueError("model_type must be 'rf', 'xgb', or 'stacking'")
    
    return model