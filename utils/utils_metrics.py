import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score




# Get predicted and true labels

def get_ys(test_ds, base_model):

    y_pred = []
    y_true = []

    for x, y in test_ds:
        predictions = base_model.predict(x)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(y.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return y_pred, y_true


# Confusion matrix visualization

def visualize_confusion_matrix(y_pred, y_true):
    cm = confusion_matrix(y_pred, y_true, normalize="true")
    plt.figure(figsize=(16, 14))
    class_names = ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", 
                   "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree",
                   "two", "up", "visual", "wow", "yes", "zero"]
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.savefig('imgs/confusion_matrix.png')
 #   plt.show()


# Precision, Recall, F1-score
    
def metrics_evaluation(y_pred, y_true, model_name):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted') 
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Create a DataFrame for easy visualization
    metrics_df = pd.DataFrame({
        'Accuracy' : [accuracy],
        'Precision' : [precision],
        'Recall' : [recall],
        'F1-score' : [f1],
        }, index=[model_name])


    # Display the table
    print(metrics_df)



def plot_history(history, columns=['loss', 'sparse_categorical_accuracy']):
    """
    Plot training history after model has been trained.
    
    Parameters:
    - history: History object returned by model.fit()
    - columns: List of metrics to plot (default: ['loss', 'sparse_categorical_accuracy'])
    """
    # Create subplots
    if len(columns) > 1:
        fig, axes = plt.subplots(len(columns), 1, figsize=(8, 5*len(columns)))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        axes = [axes]  # Make it a list for consistent indexing
    
    for i, column in enumerate(columns):
        if column not in history.history:
            print(f"Warning: '{column}' not found in history. Available metrics: {list(history.history.keys())}")
            continue
            
        ax = axes[i]
        ax.plot(history.history[column], label='training', color='blue', linewidth=1.5)
        
        val_column = 'val_'+column
        if val_column in history.history:
            ax.plot(history.history[val_column], label='validation', color='firebrick', linewidth=1.5)
        
        ax.set_xticks(range(len(history.history[column])))
        ax.set_xticklabels(range(1, len(history.history[column])+1))
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.5)
        ax.set_ylabel(column)
        ax.legend(edgecolor='black', facecolor='linen', fontsize=12, loc='best')

    plt.tight_layout()
    plt.savefig("imgs/history_plot.png")
  #  plt.show()
