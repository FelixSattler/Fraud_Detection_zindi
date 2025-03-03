import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, classification_report



def plot_confusion_matrix(y_test, y_pred, labels=["not fraud", "fraud"], values='perc'):
    """
    Plots a confusion matrix using seaborn's heatmap.

    Parameters:
    y_test (array-like): True labels of the test dataset.
    y_pred (array-like): Predicted labels by the classifier.
    labels (list, optional): List of labels for the confusion matrix axes. Defaults to ["not fraud", "fraud"].
    values (str, optional): Type of values to display in the confusion matrix. 
                            'perc' for percentage values, 'abs' for absolute values.
                            Defaults to 'perc'.

    Returns:
    None

    Displays:
    A heatmap representing the confusion matrix, either in percentages or absolute values.
    """
    cm = confusion_matrix(y_test, y_pred)
    if values == 'perc':
        cm_display = (cm / cm.sum()) * 100
        fmt = "0.2f"
        title = "Confusion Matrix (%)"
    else:  # 'abs'
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix (absolute values)"
        
    cm_display_df = pd.DataFrame(cm_display, index=labels, columns=labels)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm_display_df, annot=True, fmt=fmt, cmap="coolwarm", linewidths=0.5, linecolor="gray")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.show()

#--------------------------------------------------------------------------------------------------

def best_threshold_for_metric(X_test, y_test, model, metric):
    """
    Finds the best threshold for maximizing a specified metric.

    Parameters:
    X_test (array-like): Test dataset features.
    y_test (array-like): True labels of the test dataset.
    model (estimator): Trained model with predict_proba method.
    metric (str): The metric to maximize. Must be one of 'accuracy', 'f1', 'precision', 'recall'.

    Returns:
    float: The best threshold for maximizing the specified metric.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    
    if metric == 'f1':
        scores = [f1_score(y_test, y_probs > thr) for thr in thresholds]
    elif metric == 'precision':
        scores = [precision_score(y_test, y_probs > thr) for thr in thresholds]
    elif metric == 'recall':
        scores = [recall_score(y_test, y_probs > thr) for thr in thresholds]
    elif metric == 'accuracy':
        scores = [accuracy_score(y_test, y_probs > thr) for thr in thresholds]
    else:
        raise ValueError("Metric must be one of 'f1', 'precision', 'recall', or 'accuracy'")
    
    best_threshold = thresholds[scores.index(max(scores))]
    return best_threshold

# Example usage
# best_threshold = best_threshold_for_metric(X_test, y_test, model, metric='f1')

#--------------------------------------------------------------------------------------------------

def final_pred(model, X_test, y_test, best_threshold):
    """
    Uses the best threshold to make final predictions and prints the AUC and classification report.

    Parameters:
    model (estimator): Trained model with predict_proba method.
    X_test (array-like): Test dataset features.
    y_test (array-like): True labels of the test dataset.
    best_threshold (float): The threshold to use for making final predictions.

    Returns:
    tuple: (y_pred_best, auc) where:
        - y_pred_best (array-like): Final predictions made using the best threshold.
        - auc (float): AUC score.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred_best = (y_probs > best_threshold).astype(int)

    auc = roc_auc_score(y_test, y_probs)
    print("AUC:\n", auc, "\n")

    print("Classification Report:\n", classification_report(y_test, y_pred_best))

    return y_pred_best, auc  

# Example usage
#y_pred_best, auc = final_pred(best_model, X_test, y_test, best_threshold)
