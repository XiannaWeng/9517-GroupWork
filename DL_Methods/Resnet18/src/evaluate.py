import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
import pandas as pd
import os

def calculate_metrics(y_true, y_pred, class_names):
    """Calculates and returns common classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted') # Use weighted for overall score

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classification_report": report # Contains per-class metrics
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None):
    """Plots and optionally saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_training_history(history, title='Training History', save_path=None):
    """Plots training and validation loss/accuracy."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Loss
    axs[0].plot(history['train_loss'], label='Train Loss')
    axs[0].plot(history['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Accuracy
    axs[1].plot(history['train_acc'], label='Train Accuracy')
    axs[1].plot(history['val_acc'], label='Validation Accuracy')
    axs[1].set_title('Accuracy over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    if save_path:
         os.makedirs(os.path.dirname(save_path), exist_ok=True)
         plt.savefig(save_path)
         print(f"Training history plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    # Example Usage (assuming you have results from a run)
    # Replace with actual data
    mock_y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    mock_y_pred = [0, 1, 1, 0, 1, 2, 2, 1, 2, 0]
    mock_class_names = ['ClassA', 'ClassB', 'ClassC']
    mock_history = {
        'train_loss': [1.0, 0.8, 0.6], 'train_acc': [0.5, 0.6, 0.7],
        'val_loss': [0.9, 0.7, 0.5], 'val_acc': [0.55, 0.65, 0.75]
    }

    metrics = calculate_metrics(mock_y_true, mock_y_pred, mock_class_names)
    print("Calculated Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
    print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
    print("\nClassification Report (dict):")
    # print(metrics['classification_report']) # Can be verbose
    print(pd.DataFrame(metrics['classification_report']).transpose())


    plot_confusion_matrix(mock_y_true, mock_y_pred, mock_class_names, title='Mock CM', save_path='../results/mock_cm.png')
    plot_training_history(mock_history, title='Mock Training', save_path='../results/mock_history.png')