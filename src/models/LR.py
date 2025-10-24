import pickle
from sklearn.linear_model import LinearRegression

from src.data.preprocessing import get_xy
from src.utils.file_operations import ensure_dir

class LinearRegressor():

    def model_training(self, df_data, experiment, results_directory):
        # Created formatted target column name
        target_column_formatted = f'{experiment.target_column}_t+{experiment.lead_time}'

        # Create X and y
        df_train = df_data.copy()
        feature_columns  = df_train.drop(target_column_formatted, axis=1).columns
        X_train, y_train = get_xy(df_train, feature_columns, target_column_formatted)

        # Fit model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save model to pickle file
        model_directory = results_directory + 'models/'
        ensure_dir(model_directory)
        model_file_path = model_directory + 'hypermodel.pkl'

        with open(model_file_path, 'wb') as file:
            pickle.dump(model, file)

        return model


    def predict(self, model, X):
        # Build dictionary to save predictions to a file later
        results = {
            'predictions': model.predict(X)
        }

        return results