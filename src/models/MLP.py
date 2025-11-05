from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Dropout
import keras.metrics as km
from keras.models import Sequential, save_model, load_model
from keras.optimizers import Adam
from keras_tuner import GridSearch, Objective

from functools import partial
import numpy as np
import pandas as pd
import tensorflow as tf
import time

from src.data.preprocessing import create_kfolds_keras, get_train_test_split
import src.evaluation.metrics as m
from src.utils.file_operations import ensure_dir

class MLPRegressor():

    # Scores that need to be maximized, not minimized
    metrics_to_max = ['r2_score', 'mean_absolute_percentage_error']

    def build_model(self, hp=None, params=None, loss_function = 'mean_squared_error'):
        """
        Builds a Keras Sequential model.
        
        Parameters:
            hp: HyperParameters object (used during Keras Tuner search)
            params: dict of hyperparameters (used during manual model build)

        One of hp or params must be provided.
        """
        if hp is not None:
            num_layers = hp.Choice('num_layers', (1, 2, 3))
            neurons = hp.Choice('neurons', (50, 100, 200))
            learning_rate = hp.Choice('lr', (0.1, 0.01, 1e-3, 1e-4, 1e-5))
        elif params is not None:
            num_layers = params['num_layers']
            neurons = params['neurons']
            learning_rate = params['lr']
        else:
            raise ValueError("Either 'hp' or 'params' must be provided.")

        model = Sequential()

        for i in range(num_layers):
            # Hidden Layer
            model.add(Dense(neurons, kernel_initializer='he_normal', activation='relu'))

            # Dropout
            model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(1, activation='linear'))

        if loss_function == 'weighted_mse':
            loss_function = m.weighted_mse(0.5, 20) # GWLs > 0.5m have 20x more weight

        # Compile
        model.compile(
            loss=loss_function,
            optimizer=Adam(learning_rate=learning_rate),
            metrics=[
                km.MeanSquaredError(), km.RootMeanSquaredError(), km.MeanAbsoluteError(),
                km.MeanAbsolutePercentageError(), km.R2Score()
            ]
        )

        return model

    
    def model_training(self, df_data, experiment, results_directory):
        # Set objective metric and direction
        direction = 'min'
        objective = experiment.objective_metric

        if objective in self.metrics_to_max:
            direction = 'max'

        target_column_formatted = f'{experiment.target_column}_t+{experiment.lead_time}'

        # Create folds for k-fold validation
        kfolds = create_kfolds_keras(df_data, experiment.train_years, target_column_formatted)
        
        # Perform k-fold cross-validation and return metrics for all folds
        cv_metrics = self.kfold_cross_validation(kfolds, results_directory, loss_function = experiment.loss_function,
                                                 objective = f'val_{objective}', direction = direction)

        # Find the trial ID with the best metric score on average over all folds
        best_trial_id = self.find_best_trial_id(cv_metrics, metric = f'val_{objective}', direction = direction)

        # Find all rows with the same trial ID to extract the metrics of best hyperparameters
        df_best_trial = cv_metrics[cv_metrics['trial_id'] == best_trial_id].copy()

        # Save metrics with best trial ID to csv
        tuning_results_directory = results_directory + 'tuning/'
        cv_metrics_path = tuning_results_directory + 'best_cv_metrics.csv'
        df_best_trial.to_csv(cv_metrics_path)

        # Extract best hyperparameters from first row of the best metrics df
        best_hyperparams = df_best_trial.loc[df_best_trial.index[0], ['num_layers', 'neurons', 'lr']].to_dict()

        # Train final model ensemble using full training set and best hyperparameters
        model_ensemble = self.train_final_models(df_data, experiment, best_hyperparams, results_directory)

        return model_ensemble
    

    def kfold_cross_validation(self, kfolds, results_directory, loss_function = 'mean_squared_error',
                               objective = 'val_mean_squared_error', direction = 'min'):
        tuning_results_directory = results_directory + 'tuning/'

        df_metrics_list = []
        for fold in kfolds:
            valid_years = fold['valid_years']
            X_train, y_train = fold['train_data']
            X_valid, y_valid = fold['valid_data']

            # Set the project folder name as the validation year
            tuning_project_name = str(valid_years[0])

            print(f'Fold Validation Year: {valid_years[0]}')

            start = time.time()
            df_metrics = self.tune_model(X_train, y_train, X_valid, y_valid, tuning_results_directory, 
                                         tuning_project_name, loss_function = loss_function, 
                                         objective = objective, direction = direction)
            end = time.time()

            print(f"Tuning took {(end - start)/60:.2f} minutes")

            # Add validation year to metrics df for this fold
            df_metrics['validation_years'] = str(valid_years[0])

            # Add metrics for this fold to full list of metrics
            df_metrics_list.append(df_metrics)
            
        df_all_metrics = pd.concat(df_metrics_list, ignore_index=True)
        
        return df_all_metrics
        
    
    def tune_model(self, x_train, y_train, x_val, y_val, directory, project_name,
                 loss_function = 'mean_squared_error', objective='val_mean_squared_error', direction='min',
                 batch_size=64, validation_batch_size=64, epochs=2000, 
                 patience=20):
        """
        Hyperparameter tuning using GridSearch.
        
        Args:
            x_train, y_train: training data (NumPy arrays).
            x_val, y_val: validation data (NumPy arrays).
            directory: folder where results will be saved.
            project_name: project identifier for KerasTuner.
            objective: metric to optimize (default: val_mean_squared_error).
            direction: "min" or "max".
            batch_size: training batch size.
            validation_batch_size: validation batch size.
            epochs: training epochs per trial.
            patience: early stopping patience.
        """

        self.tuner = GridSearch(
            hypermodel=partial(self.build_model, loss_function = loss_function),
            objective=Objective(objective, direction=direction),
            seed=42,
            directory=directory,
            project_name=project_name,
            overwrite=False
        )

        early_stopping = EarlyStopping(
            monitor=objective,
            patience=patience,
            mode=direction,
            restore_best_weights=True,
            verbose=0
        )

        tensor_board_logs_path = directory + f'{project_name}/tb_logs/'
        tensor_board = TensorBoard(tensor_board_logs_path, update_freq='epoch')

        callbacks = [early_stopping, tensor_board]

        self.tuner.search(
            x_train, y_train,
            epochs = epochs,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            validation_batch_size=validation_batch_size,
            callbacks=callbacks,
            verbose=0
        )

        df_metrics = self.save_trial_data()

        print(self.tuner.results_summary())

        return df_metrics


    def save_trial_data(self):
        trial_data = []

        for trial in self.tuner.oracle.trials.values():
            entry = {
                "trial_id": trial.trial_id,
                "status": trial.status,
                **trial.hyperparameters.values
            }

            # Find the epoch with the best score based on the given objective metric for this trial/
            # This epoch is where the model weights are saved after early stopping kicks in
            best_epoch = trial.best_step
        
            # Add it to the entry dictionary
            entry["best_epoch"] = best_epoch

            # Find and save all metric values at the best epoch from above
            for metric_name, metric in trial.metrics.metrics.items():
                best_observation = metric._observations.get(best_epoch)

                entry[metric_name] = best_observation.value[0]

            trial_data.append(entry)

        df = pd.DataFrame(trial_data)
        
        return df


    def find_best_trial_id(self, df_metrics, metric = 'val_mean_squared_error', direction = 'min'):
        # Group by trial ID and calculate mean of metric
        df_grouped = df_metrics.groupby('trial_id')[metric].mean().reset_index()

        # Find the trial ID with the best mean metric score over all folds
        if direction == 'max':
            best_trial_id = df_grouped.loc[df_grouped[metric].idxmax(), 'trial_id']
        else:
            best_trial_id = df_grouped.loc[df_grouped[metric].idxmin(), 'trial_id']

        return best_trial_id
    

    def train_final_models(self, df_data, experiment, best_hyperparams, results_directory):
        # Use 2023 as validation year for early stopping
        X_train, y_train, X_valid, y_valid = get_train_test_split(df_data, experiment, [2023])

        model_directory = results_directory + 'models/'
        ensure_dir(model_directory)

        direction = 'min'
        objective = experiment.objective_metric

        if objective in self.metrics_to_max:
            direction = 'max'

        start = time.time()
        
        # Train ensemble models using best hyperparameters
        ensemble_models = self.train_ensemble(X_train, y_train, X_valid, y_valid, best_hyperparams, 
                                              model_directory, loss_function = experiment.loss_function, 
                                              objective = f'val_{objective}', direction = direction)
        end = time.time()

        print(f"Ensemble training took {(end - start)/60:.2f} minutes")

        return ensemble_models
    

    def train_ensemble(self, X_train, y_train, X_valid, y_valid, best_hyperparams, model_directory, n_models=30,
                       loss_function = 'mean_squared_error', objective = 'val_mean_squared_error', direction = 'max', 
                       epochs = 10000, batch_size = 64, validation_batch_size = 64, patience = 20):
        models = []
        for i in range(n_models):
            model_file_path = model_directory + f'hypermodel{i+1}.h5'

            try:
                model = load_model(model_file_path)
            except OSError:
                # Set a different seed per model for weight initialization diversity
                tf.random.set_seed(i)

                # Build model with optimal hyperparameters from k-fold cross-validation
                model = self.build_model(params = best_hyperparams, loss_function = loss_function)

                # Define callbacks
                early_stopping = EarlyStopping(monitor = f'{objective}', mode = direction, patience = patience, 
                                            restore_best_weights = True, verbose = 0)

                callbacks = [early_stopping]

                model.fit(
                    X_train, y_train,
                    validation_data = (X_valid, y_valid),
                            epochs = epochs,
                            batch_size = batch_size,
                            validation_batch_size = validation_batch_size,
                            callbacks = callbacks,
                            verbose = 0
                            )
                
                save_model(model, model_file_path)
            
            models.append(model)

        return models
    

    def predict(self, models, X):
        # Make predictions from each model in the ensemble
        ensemble_preds = np.array([model.predict(X, verbose=0).flatten() for model in models])

        # Take the median of the ensemble predictions
        ensemble_median = np.median(ensemble_preds, axis=0)

        # Build dictionary to save predictions to a file later
        results = {
            'predictions': ensemble_median
        }

        # Add predictions for each ensemble member to dictionary
        for i, preds in enumerate(ensemble_preds, start=1):
            results[f'model_{i}'] = preds

        return results