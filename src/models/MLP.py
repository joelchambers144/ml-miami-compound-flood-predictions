from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Dropout
import keras.metrics as km
from keras.models import Sequential, save_model, load_model
from keras.optimizers import Adam
from keras_tuner import Hyperband, Objective

import numpy as np
import pandas as pd
import tensorflow as tf

import src.evaluation.metrics as m 
from src.data.preprocessing import create_partitions_keras, get_train_test_split
from src.utils.file_operations import ensure_dir

class MLPRegressor():

    def build_model(self, hp=None, params=None):
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
        model.add(Dense(1, activation='sigmoid'))

        # Compile
        model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[
            km.MeanSquaredError(), km.RootMeanSquaredError(), km.MeanAbsoluteError(),
            km.MeanAbsolutePercentageError(), km.R2Score()
        ]
        )

        return model

    
    def hyperparameter_tuning(self, df_data, experiment, results_directory):
        direction = 'min'
        objective = experiment.objective_metric

        if objective == 'r2_score':
            direction = 'max'

        target_column_formatted = f'{experiment.target_column}_t+{experiment.lead_time}'

        # Create partitions for k-fold validation
        partitions = create_partitions_keras(df_data, experiment.train_years, target_column_formatted)
        
        # Perform k-fold cross-validation
        cv_metrics = self.kfold_cross_validation(partitions, results_directory, objective = f'val_{objective}', direction = direction)

        # Find the trial ID with the best metric score
        best_trial_id = self.find_best_trial_id(cv_metrics, metric = f'val_{objective}', direction = direction)

        # Find all rows with the same trial ID to extract the metrics of best hyperparameters
        df_best_trial = cv_metrics[cv_metrics['trial_id'] == best_trial_id].copy()

        # Calculate post-training evaluation metrics row-wise (for all validation years)
        #df_best_trial = self.calculate_post_train_metrics(df_best_trial)

        # Save metrics with best trial ID to csv
        tuning_results_directory = results_directory + 'tuning/'
        cv_metrics_path = tuning_results_directory + 'best_cv_metrics.csv'
        df_best_trial.to_csv(cv_metrics_path)

        # Extract best hyperparameters from first row of the best metrics df
        best_hyperparams = df_best_trial.loc[df_best_trial.index[0], ['num_layers', 'neurons', 'lr']].to_dict()

        return best_hyperparams
    

    def kfold_cross_validation(self, partitions, results_directory, objective = 'val_mean_squared_error', direction = 'min'):
        tuning_results_directory = results_directory + 'tuning/'

        df_metrics_list = []
        for partition in partitions:
            train_years = partition['train_years']
            valid_years = partition['valid_years']
            X_train, y_train = partition['train_data']
            X_valid, y_valid = partition['valid_data']

            # Set the project folder name as the validation year
            tuning_project_name = str(valid_years[0])
            import time
            start = time.time()
            df_metrics = self.tune_model(X_train, y_train, X_valid, y_valid, 
                                          tuning_results_directory, tuning_project_name,
                                          objective = objective, direction = direction)

            end = time.time()

            print(f"Tuning took {(end - start)/60:.2f} minutes")
            df_metrics['validation_years'] = str(valid_years[0])
            df_metrics_list.append(df_metrics)
            
        df_all_metrics = pd.concat(df_metrics_list, ignore_index=True)
        
        return df_all_metrics
        

    def tune_model(self, x_train, y_train, x_val, y_val, directory, project_name,
                    objective='val_mean_squared_error', direction='min',
                    batch_size=64, validation_batch_size=64,
                    max_epochs=300, patience=10, factor=3):
        """
        Hyperparameter tuning using Hyperband.

        Parameters:
            x_train, y_train: training data (NumPy arrays)
            x_val, y_val: validation data (NumPy arrays)
            directory: folder where results will be saved
            project_name: project identifier for KerasTuner
            objective: metric to optimize (default: val_mean_squared_error)
            direction: "min" or "max"
            batch_size: training batch size
            validation_batch_size: validation batch size
            max_epochs: maximum training epochs per trial
            patience: early stopping patience
            factor: reduction factor for Hyperband (controls aggressiveness)
        """

        self.tuner = Hyperband(
            hypermodel=self.build_model,
            objective=Objective(objective, direction=direction),
            max_epochs=max_epochs,
            factor=factor,                # higher = fewer models trained deeply
            seed=42,
            directory=directory,
            project_name=project_name,
            overwrite=False               # reload past trials if present
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
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            validation_batch_size=validation_batch_size,
            callbacks=callbacks,
            verbose=0
        )

        df_metrics = self.save_trial_data()

        return df_metrics

        
    def save_trial_data(self):
        trial_data = []

        for trial in self.tuner.oracle.trials.values():
            entry = {
                "trial_id": trial.trial_id,
                "status": trial.status,
                **trial.hyperparameters.values
            }
            
            for metric_name, metric in trial.metrics.metrics.items():
                entry[metric_name] = metric.get_best_value()
            
            trial_data.append(entry)

        df = pd.DataFrame(trial_data)
        
        return df


    def find_best_trial_id(self, df_metrics, metric = 'val_mean_squared_error', direction = 'min'):
        # Group by trial ID and calculate mean of metric
        df_grouped = df_metrics.groupby('trial_id')[metric].mean().reset_index()

        # Find the trial ID with the best metric score
        if direction == 'max':
            best_trial_id = df_grouped.loc[df_grouped[metric].idxmax(), 'trial_id']
        else:
            best_trial_id = df_grouped.loc[df_grouped[metric].idxmin(), 'trial_id']

        return best_trial_id
    

    def calculate_post_train_metrics(self, df_best_trial):
        df_best_trial['CSI'] = df_best_trial.apply(lambda row: m.csi(row['false_positives'], row['false_negatives'], row['true_positives']), axis=1)
        df_best_trial['FAR'] = df_best_trial.apply(lambda row: m.far(row['false_positives'], row['true_positives']), axis=1)
        df_best_trial['POD'] = df_best_trial.apply(lambda row: m.pod(row['false_negatives'], row['true_positives']), axis=1)
        df_best_trial['PSS'] = df_best_trial.apply(lambda row: m.pss(row['true_negatives'], row['false_positives'], row['false_negatives'], row['true_positives']), axis=1)
        df_best_trial['HSS'] = df_best_trial.apply(lambda row: m.hss(row['true_negatives'], row['false_positives'], row['false_negatives'], row['true_positives']), axis=1)
        df_best_trial['val_CSI'] = df_best_trial.apply(lambda row: m.csi(row['val_false_positives'], row['val_false_negatives'], row['val_true_positives']), axis=1)
        df_best_trial['val_FAR'] = df_best_trial.apply(lambda row: m.far(row['val_false_positives'], row['val_true_positives']), axis=1)
        df_best_trial['val_POD'] = df_best_trial.apply(lambda row: m.pod(row['val_false_negatives'], row['val_true_positives']), axis=1)
        df_best_trial['val_PSS'] = df_best_trial.apply(lambda row: m.pss(row['val_true_negatives'], row['val_false_positives'], row['val_false_negatives'], row['val_true_positives']), axis=1)
        df_best_trial['val_HSS'] = df_best_trial.apply(lambda row: m.hss(row['val_true_negatives'], row['val_false_positives'], row['val_false_negatives'], row['val_true_positives']), axis=1)

        return df_best_trial
    

    def train_final_model(self, df_data, experiment, best_hyperparams, results_directory):
        # Use 2023 as validation year for early stopping
        X_train, y_train, X_valid, y_valid = get_train_test_split(df_data, experiment, [2023])

        model_directory = results_directory + 'models/'
        ensure_dir(model_directory)

        direction = 'min'
        objective = experiment.objective_metric

        if objective == 'r2_score':
            direction = 'max'

        # Train ensemble models using best hyperparameters
        ensemble_models = self.train_ensemble(X_train, y_train, X_valid, y_valid, best_hyperparams, 
                                              model_directory, objective = f'val_{objective}', direction = direction)
        
        '''
        # Find best decision threshold based on defined metric
        y_probs_valid = self.predict(ensemble_models, X_valid)
        best_threshold, best_score, all_scores = find_best_decision_threshold(y_valid, y_probs_valid, 
                                                                                   metric=experiment.threshold_metric)
        '''

        return ensemble_models
    

    def train_ensemble(self, X_train, y_train, X_valid, y_valid, best_hyperparams, model_directory, n_models=5,
                       objective = 'val_mean_squared_error', direction = 'max', epochs = 10000, batch_size = 64, 
                       validation_batch_size = 64, patience = 20):
        models = []
        for i in range(n_models):
            model_file_path = model_directory + f'hypermodel{i+1}.h5'

            try:
                model = load_model(model_file_path)
            except OSError:
                # Set a different seed per model for weight initialization diversity
                tf.random.set_seed(i)

                # Build model with optimal hyperparameters from k-fold cross-validation
                model = self.build_model(params = best_hyperparams)

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

        # Average them (soft voting)
        ensemble_mean = np.mean(ensemble_preds, axis=0)

        return ensemble_mean