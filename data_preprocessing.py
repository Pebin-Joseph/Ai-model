import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    print(data.head())
    return data

# Preprocess dataset
def preprocess_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
