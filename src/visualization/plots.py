"""
Visualization utilities for hotel reservations analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics


def plot_feature_distributions(df, num_cols, cat_cols):
    """
    Plot distributions for numerical and categorical features.
    
    Args:
        df: DataFrame with features
        num_cols: List of numerical column names
        cat_cols: List of categorical column names
    """
    # Plot numerical features
    fig, axes = plt.subplots(len(num_cols)//2 + len(num_cols)%2, 2, figsize=(15, len(num_cols)*3))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        df[col].hist(bins=30, ax=axes[i], alpha=0.7)
        axes[i].set_title(f'{col} Distribution')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for i in range(len(num_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Plot categorical features
    for cat_feature in cat_cols:
        plt.figure(figsize=(10, 6))
        df[cat_feature].value_counts().plot(kind='bar', color='skyblue')
        plt.title(f'{cat_feature} Distribution')
        plt.xlabel(cat_feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()


def plot_target_distribution(df, target_col='booking_status'):
    """
    Plot the distribution of the target variable.
    
    Args:
        df: DataFrame
        target_col: Name of target column
    """
    plt.figure(figsize=(8, 6))
    df[target_col].value_counts().plot(kind='bar', color=['lightblue', 'lightcoral'])
    plt.title(f'{target_col} Distribution')
    plt.xlabel('Booking Status')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()
    
    # Print percentages
    percentages = df[target_col].value_counts(normalize=True) * 100
    print(f"Target distribution:")
    for status, pct in percentages.items():
        print(f"  {status}: {pct:.2f}%")


def plot_box_target(df, target, num_features):
    """
    Create box plots for numerical features vs target variable.
    
    Args:
        df: DataFrame
        target: Target column name
        num_features: List of numerical feature names
    """
    num_plots = len(num_features)
    num_rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
    axes = axes.flatten()
    
    for i, column in enumerate(num_features):
        sns.boxplot(x=target, y=column, ax=axes[i], data=df, palette="Blues")
        axes[i].set_title(f'{column} vs {target}')
    
    # Hide empty subplots
    for i in range(len(num_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_cat_features_with_target(df, target, cat_features):
    """
    Plot categorical features vs target variable.
    
    Args:
        df: DataFrame
        target: Target column name
        cat_features: List of categorical feature names
    """
    num_features = len(cat_features)
    num_row = (num_features + 1) // 2
    
    fig, axes = plt.subplots(num_row, 2, figsize=(12, num_row * 5))
    axes = axes.flatten()
    
    for i, feature in enumerate(cat_features):
        if feature != target:
            crosstab = pd.crosstab(df[feature], df[target], normalize='index')
            crosstab.plot(kind='bar', ax=axes[i], stacked=True, 
                         color=['lightblue', 'lightcoral'])
            axes[i].set_title(f'{feature} vs {target}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Proportion')
            axes[i].legend(title=target)
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(len(cat_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, num_cols):
    """
    Plot correlation matrix for numerical features.
    
    Args:
        df: DataFrame
        num_cols: List of numerical column names
    """
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[num_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()


def plot_price_by_month(df):
    """
    Plot average price by arrival month.
    
    Args:
        df: DataFrame with price and arrival month data
    """
    plt.figure(figsize=(10, 6))
    if 'avg_price_per_room' in df.columns and 'arrival_month' in df.columns:
        sns.barplot(x='arrival_month', y='avg_price_per_room', data=df)
        plt.xlabel('Month')
        plt.ylabel('Average Price Per Room')
        plt.title('Average Price Per Room by Month')
        plt.show()


def plot_confusion_matrix(model, X_test, y_test, labels=[0, 1]):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        labels: Class labels
    """
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)
    
    # Convert confusion matrix to DataFrame
    df_cm = pd.DataFrame(cm, index=["Actual - No", "Actual - Yes"], 
                         columns=["Predicted - No", "Predicted - Yes"])
    
    # Calculate percentages
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels_formatted = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels_formatted = np.asarray(labels_formatted).reshape(2, 2)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=labels_formatted, fmt='', cmap='Blues', 
                cbar=False, annot_kws={"size": 16})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC curve for model predictions.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_precision_recall_curve(model, X_test, y_test):
    """
    Plot Precision-Recall curve for model predictions.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_prob)
    average_precision = metrics.average_precision_score(y_test, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2, 
             label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not have feature_importances_ attribute")


def plot_model_comparison(results_df):
    """
    Plot comparison of model performance metrics.
    
    Args:
        results_df: DataFrame with model evaluation results
    """
    metrics_to_plot = ['test_accuracy', 'test_f1', 'test_roc_auc']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics_to_plot):
        results_df[metric].plot(kind='bar', ax=axes[i], color='skyblue')
        axes[i].set_title(f'Model Comparison - {metric.replace("test_", "").upper()}')
        axes[i].set_ylabel(metric.replace("test_", "").replace("_", " ").title())
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_interactive_correlation_heatmap(df, num_cols):
    """
    Create an interactive correlation heatmap using Plotly.
    
    Args:
        df: DataFrame
        num_cols: List of numerical column names
    """
    correlation_matrix = df[num_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdYlBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Interactive Correlation Matrix',
        width=800,
        height=800
    )
    
    fig.show()


def plot_evaluation_metrics(model, X_test, y_test):
    """
    Plot all evaluation metrics for a model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    """
    print("Plotting model evaluation metrics...")
    
    # Plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test)
    
    # Plot ROC curve
    plot_roc_curve(model, X_test, y_test)
    
    # Plot Precision-Recall curve
    plot_precision_recall_curve(model, X_test, y_test)