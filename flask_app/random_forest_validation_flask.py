import pandas as pd
import requests
import random
import os

def load_data(data_directory, file_name):
    file_path = os.path.join(data_directory, file_name)
    return pd.read_csv(file_path)

def send_prediction_request(X, y, num_records):
    max_records = len(X)
    num_records = min(num_records, max_records)
    indices = random.sample(range(max_records), num_records)

    features = X.iloc[indices].values.tolist()
    y_validation = y.iloc[indices].values.tolist() if y is not None else None

    url = 'http://127.0.0.1:5000/predict'
    data = {'features': features, 'y_validation': y_validation}
    response = requests.post(url, json=data)

    return response.json()

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/processed_data'))
    X_valid = load_data(data_dir, 'X_valid_selected.csv')
    y_valid = load_data(data_dir, 'y_valid.csv')

    num_records = int(input("Enter the number of records to predict: "))
    result = send_prediction_request(X_valid, y_valid, num_records)
    print(result)
