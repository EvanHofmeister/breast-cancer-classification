import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_save_data(data_file_path, base_data_directory, data_subdirectory):
    feature_order = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                     'compactness', 'concavity', 'concave_points', 'symmetry',
                     'fractal_dimension']
    custom_column_names = ['id', 'diagnosis'] + [f'{feature}_{i}' for i in range(1, 4) for feature in feature_order]

    df = pd.read_csv(data_file_path, header=None, names=custom_column_names)
    df = df.set_index('id')

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis'].map({'M': 1, 'B': 0})

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    ## Create dataframes for selected features (LASSO Method used, see EDA jupyter notebook for details)
    lasso_selected_features = ['texture_1', 'area_1', 'area_2', 'radius_3', 'texture_3', 'perimeter_3', 'area_3', 'concavity_3']
    # Define new dataframes based on selected features
    X_train_selected = X_train[lasso_selected_features]
    X_valid_selected = X_valid[lasso_selected_features]
    X_test_selected = X_test[lasso_selected_features]

    dataframes = {'df': df, 'X_train': X_train, 'X_valid': X_valid, 'X_test': X_test,
                  'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
                  'X_train_selected': X_train_selected, 'X_valid_selected': X_valid_selected, 'X_test_selected': X_test_selected}

    data_subdirectory_path = os.path.join(base_data_directory, data_subdirectory)
    if not os.path.exists(data_subdirectory_path):
        os.makedirs(data_subdirectory_path)

    for df_name, df in dataframes.items():
        file_path = os.path.join(data_subdirectory_path, f'{df_name}.csv')
        df.to_csv(file_path, index=False)
    print("Data split and saved.")

if __name__ == "__main__":
    base_data_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data'))
    data_file_path = os.path.join(base_data_directory, 'wdbc.data')
    split_and_save_data(data_file_path, base_data_directory, 'processed_data')
