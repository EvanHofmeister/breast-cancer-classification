import pandas as pd
import os
from joblib import dump
from train_model import train_model
from evaluate_model import evaluate_model
from sklearn.ensemble import RandomForestClassifier

def load_data(data_directory):
    data = {}
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            data[filename.split('.')[0]] = pd.read_csv(file_path)
    return data

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/processed_data'))
    data = load_data(data_dir)
    X_train, y_train = data['X_train_selected'], data['y_train']
    X_test, y_test = data['X_test_selected'], data['y_test']

    rf_params = {
        # Define your parameter grid
    }
    rf_model, rf_best_params = train_model(RandomForestClassifier(), rf_params, X_train, y_train, cv_folds=5)

    results_dict = {}
    evaluate_model(rf_model, X_test, y_test, "Random Forest", results_dict)

    model_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'model'))

    os.makedirs(model_dir, exist_ok=True)
    joblib_file = os.path.join(model_dir, "rf_model.joblib")
    dump(rf_model, joblib_file)
