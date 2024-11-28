import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted Values")
    plt.show()
