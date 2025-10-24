import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as skm
from sklearn.model_selection import GridSearchCV
import time

from src.data.preprocessing import get_xy, create_partitions_sklearn
from src.utils.file_operations import ensure_dir

class RFRegressor():

    loss_map = {'mean_squared_error': 'squared_error'}

    def tune_model(self, X, y, custom_folds, results_directory, loss_function = 'mean_squared_error',
                   objective = 'mean_squared_error'):
        param_grid = {
            'n_estimators': [100, 200, 1000],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }

        scoring = {
            'mean_squared_error': skm.make_scorer(skm.mean_squared_error),
            'root_mean_squared_error': skm.make_scorer(skm.root_mean_squared_error),
            'mean_absolute_error': skm.make_scorer(skm.mean_absolute_error),
            'mean_absolute_percentage_error': skm.make_scorer(skm.mean_absolute_percentage_error),
            'r2_score': skm.make_scorer(skm.r2_score)
        }

        # Map passed experiment loss function to sklearn equivalent
        formatted_loss_function = self.loss_map[loss_function]

        if objective == 'loss':
            objective = loss_function

        tuner = GridSearchCV(
            RandomForestRegressor(criterion=formatted_loss_function, random_state=42, verbose = 0),
            param_grid,
            cv=custom_folds,
            scoring=scoring,
            refit=objective,
            return_train_score=True,
            n_jobs=None,
            verbose=0
        )

        # Calculate and print how long tuning takes
        start = time.time()
        tuner.fit(X, y)
        end = time.time()
        print(f"Tuning took {(end - start)/60:.2f} minutes")

        tuning_directory = results_directory + 'tuning/'
        ensure_dir(tuning_directory)

        # Save all tuning results to file
        all_tuning_results_path = tuning_directory + 'all_results.csv'
        df_results = pd.DataFrame(tuner.cv_results_)
        df_results.to_csv(all_tuning_results_path)
    
        # Find row with best tuning results based on given objective metric
        best_row = df_results.loc[[tuner.best_index_]]

        # Save best results to file
        best_tuning_results_path = tuning_directory + 'best_results.csv'
        best_row.to_csv(best_tuning_results_path)

        print(f'Best tuning trial:', best_row)

        return tuner.best_estimator_
    

    def model_training(self, df_data, experiment, results_directory):
        # Set objective metric
        objective = experiment.objective_metric

        # Create formatted target column name
        target_column_formatted = f'{experiment.target_column}_t+{experiment.lead_time}'

        # Create X and y
        df_train = df_data.copy()
        feature_columns  = df_train.drop(target_column_formatted, axis=1).columns
        X_train, y_train = get_xy(df_train, feature_columns, target_column_formatted)

        # Create folds for k-fold validation
        custom_folds = create_partitions_sklearn(df_train, experiment.train_years)

        # Perform k-fold cross-validation and return best model
        best_model = self.tune_model(X_train, y_train, custom_folds, results_directory,
                                           experiment.loss_function, objective)
        
        # Save model to pickle file
        model_directory = results_directory + 'models/'
        ensure_dir(model_directory)
        model_file_path = model_directory + 'hypermodel.pkl'

        with open(model_file_path, 'wb') as file:
            pickle.dump(best_model, file)

        return best_model


    def predict(self, model, X):
        # Build dictionary to save predictions to a file later
        results = {
            'predictions': model.predict(X)
        }

        return results