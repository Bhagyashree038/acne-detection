import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save the confusion matrix plot as a PNG image
    plt.savefig("confusion_matrix.png")  # Save the plot as an image file
    print("Confusion Matrix plot saved as confusion_matrix.png")
    
    # Show the plot
    plt.show()

# Function to plot and save accuracy graph
def plot_accuracy_graph(accuracy_list):
    plt.plot(accuracy_list)
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Save the accuracy graph as a PNG image
    plt.savefig("accuracy_graph.png")  # Save the plot as an image file
    print("Accuracy graph saved as accuracy_graph.png")
    
    # Show the plot
    plt.show()

# Example usage
# Assume you're using a dataset where each sample has a true label (0 for clear skin, 1 for mild acne, 2 for severe acne)
# Replace `dataset_true_labels` with the actual ground truth labels from your test data
y_true = [0, 1, 2, 0, 1]  # Actual labels for the test dataset
y_pred = [0, 1, 1, 0, 2]  # Predicted labels from the model
labels = [0, 1, 2]  # Labels: 0 = Clear Skin, 1 = Mild Acne, 2 = Severe Acne

# # Call function to plot and save confusion matrix
# plot_confusion_matrix(y_true, y_pred, labels)

# Example accuracy list (e.g., model accuracy over different epochs)
accuracy_list = [0.85, 0.87, 0.88, 0.89, 0.90]  # Example accuracy values over epochs

# Call function to plot and save accuracy graph
plot_accuracy_graph(accuracy_list)
