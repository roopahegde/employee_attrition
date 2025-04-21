"""
Evaluation utilities for model performance assessment.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities for the positive class.
        
    Returns:
        dict: Dictionary with evaluation metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def print_metrics(metrics):
    """
    Print formatted metrics.
    
    Args:
        metrics (dict): Dictionary with evaluation metrics.
    """
    print("\nModel Performance Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")


def plot_confusion_matrix(y_true, y_pred, figsize=(8, 6)):
    """
    Plot a confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        figsize (tuple, optional): Figure size. Defaults to (8, 6).
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10, figsize=(10, 6)):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names (list): Names of the features.
        top_n (int, optional): Number of top features to show. Defaults to 10.
        figsize (tuple, optional): Figure size. Defaults to (10, 6).
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Display top features
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return feature_importance_df


def print_classification_report(y_true, y_pred):
    """
    Print a formatted classification report.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
    """
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print("-" * 60)
    print(report)