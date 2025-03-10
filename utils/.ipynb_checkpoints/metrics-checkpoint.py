import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    log_loss,
)


def calculate_metrics(true_labels, pred_prob):
    pred_labels = (pred_prob > 0.5).astype(int)
    TN, FP, FN, TP = confusion_matrix(true_labels, pred_labels).ravel()
    
    # Calculate metrics
    sensitivity = np.round((TP / (TP + FN)), 4)  # Sensitivity (Recall)
    specificity = np.round((TN / (TN + FP)), 4)  # Specificity
    auc = np.round(roc_auc_score(true_labels, pred_prob), 4)  # AUC
    accuracy = np.round(accuracy_score(true_labels, pred_labels), 4)  # Accuracy
    f1 = np.round(f1_score(true_labels, pred_labels), 4)  # F1 Score
    logloss = np.round(log_loss(true_labels, pred_prob), 4)  # Log Loss
    
    # Calculate PPV (Positive Predictive Value) and NPV (Negative Predictive Value)
    PPV = np.round((TP / (TP + FP)), 4)  # PPV = Precision
    NPV = np.round((TN / (TN + FN)), 4)  # NPV
    
    return sensitivity, specificity, auc, accuracy, f1, logloss, PPV, NPV

# Function to calculate confidence intervals
def confidence_interval(data, confidence=0.95):
    mean = np.round(np.mean(data),4)
    ci_lower = np.round(np.percentile(data, (1 - confidence) / 2 * 100),4)
    ci_upper = np.round(np.percentile(data, (1 + confidence) / 2 * 100),4)
    return ci_lower, mean, ci_upper