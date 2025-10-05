#experiment_runner.py
#----------------------------------
# Created By : Beto Estrada
#----------------------------------
""" 
This script takes an experiment object, dataset, model architecture (Multilayer Perceptron (MLP)), 
and a results directory to place the results for the specific experiment being run. From there the 
experiment pipeline is run consisting of the following steps: 

1. Split up train and test sets
2. Hyperparameter tuning paired with k-fold cross-validation on training set
3. Train final model using best hyperparameters from step 2 and full training set
4. Evaluate model on test set
""" 
#---------------------------------------------
# 
#
import pandas as pd

from src.data.preprocessing import get_xy, split_df_by_years, create_input_columns, order_input_arrays, create_lagged_columns
import src.evaluation.metrics as m
from src.models.MLP import MLPRegressor
from src.utils.file_operations import ensure_dir

def experiment_pipeline(experiment, df_data, model_architecture, results_directory):
    model = choose_model_architecture(model_architecture)
    
    # Results for individual model go into a directory the same name as the model
    model_results_directory = results_directory + f'{model_architecture}/'

    # Create lagged input columns
    df_inputs = create_input_columns(df_data, experiment.input_specifications)

    # Create target column
    df_inputs = create_lagged_columns(df_inputs, experiment.target_column, (experiment.lead_time, experiment.lead_time))

    # Drop any NaNs that were created
    df_inputs.dropna(inplace=True)

    # Order input columns by given column prefix names and in ascending order based on lead time 
    column_prefixes = [input['column'] for input in experiment.input_specifications]
    
    df_inputs_ordered = order_input_arrays(df_inputs, column_prefixes)

    # If target column is not in input specifications then target column must be added manually
    if experiment.target_column not in column_prefixes:
        df_inputs_ordered[f'{experiment.target_column}_t+{experiment.lead_time}'] = df_inputs[f'{experiment.target_column}_t+{experiment.lead_time}'].copy()

    # Split up test year from rest of data
    df_test, df_train = split_df_by_years(df_inputs_ordered, experiment.test_years)

    # Find best hyperparameters using GridSearch & k-fold cross-validation
    best_hyperparams = model.hyperparameter_tuning(df_train, experiment, model_results_directory)

    # Train final model(s) using full training set and best hyperparameters
    best_model = model.train_final_model(df_train, experiment, best_hyperparams, model_results_directory)

    # Create test results path for this model
    model_test_results_directory = model_results_directory + 'test/'
    ensure_dir(model_test_results_directory)
    test_results_path = model_test_results_directory + 'results.csv'

    # Attempt to read in existing test metrics for experiment. If they don't exist,
    # then calculate the test metrics for this experiment
    try:
        df_test_metrics = pd.read_csv(test_results_path)
    except FileNotFoundError:
        # Create test X and y
        target_column_formatted = f'{experiment.target_column}_t+{experiment.lead_time}'
        df_test_inputs = df_test.drop(target_column_formatted, axis=1)
        feature_columns = df_test_inputs.columns
        X_test, y_test = get_xy(df_test, feature_columns, target_column_formatted)
        
        # Make predictions on test set
        y_pred_test = model.predict(best_model, X_test)

        # Create test predictions path for this model
        test_predictions_path = model_test_results_directory + 'predictions.csv'

        # Add labels to prediction dictionary
        y_pred_test['labels'] = y_test

        # Save test predictions and labels to csv file to be plotted later
        df_predictions = pd.DataFrame(y_pred_test, index = df_test.index)
        df_predictions.to_csv(test_predictions_path)
        
        # Evaluate model performance on test set
        df_test_metrics = m.evaluate_model(y_test, y_pred_test['predictions'])
        df_test_metrics['model'] = model_architecture

        # Save test metrics to results folder
        df_test_metrics.to_csv(test_results_path)

    return df_test_metrics


def choose_model_architecture(model_architecture):
    if model_architecture == 'MLP':
        model = MLPRegressor()
    else:
        raise ValueError("Unsupported model architecture.")
    
    return model