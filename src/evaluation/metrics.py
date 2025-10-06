#metrics.py
#----------------------------------
# Created By : Beto Estrada
#----------------------------------
""" 
This script holds functions calculating various performance metrics for
trained ML models.
""" 
#---------------------------------------------
# 
#
import pandas as pd
import sklearn.metrics as skm
import tensorflow as tf

def weighted_mse(threshold=5.0, weight=5.0):
    def loss(y_true, y_pred):
        mask = tf.cast(y_true > threshold, tf.float32)
        sample_weights = 1.0 + (weight - 1.0) * mask
        return tf.reduce_mean(sample_weights * tf.square(y_true - y_pred))
    return loss


def calculate_central_frequency_percentage(labels, predictions, cm):
  """Find the percentage of predictions with a central frequency (CF) of less than
  or equal to a given number of centimeters (cm)

	Parameters:
        labels (array): Labels

        predictions (array): Model predictions

        cm (int): Number of centimeters

	Returns:
		(float): central frequency (CF) percentage
	"""
  less_than_cm_counter = 0

  # Convert cm to m
  cm_to_m = cm / 100

  for index, prediction in enumerate(predictions):
    if abs(labels[index] - prediction) <= cm_to_m:
      less_than_cm_counter += 1

  cf_percentage = (less_than_cm_counter / len(predictions)) * 100

  return cf_percentage


def evaluate_model(labels, predictions):
  """Calculates Central Frequency (CF), Mean Squared Error (MSE), Root Mean Squared Error(RMSE),
  Mean Absolute Error (MAE), Median Absolute Error, and R-squared (R2)

	Parameters:
        model (tf.keras.model): The trained model

        input_array (array): Input array

        labels (array): Labels

  Returns:
      (pd.DataFrame): DataFrame containing evaluation metrics
	"""
  metrics = {}
  
  # Calculate evaluation metrics
  metrics['CF_15CM'] = calculate_central_frequency_percentage(labels, predictions, 15)
  metrics['CF_5CM'] = calculate_central_frequency_percentage(labels, predictions, 5)
  metrics['CF_1CM'] = calculate_central_frequency_percentage(labels, predictions, 1)
  metrics['MSE'] = skm.mean_squared_error(labels, predictions)
  metrics['RMSE'] = skm.root_mean_squared_error(labels, predictions)
  metrics['MAE'] = skm.mean_absolute_error(labels, predictions)
  metrics['MEDAE'] = skm.median_absolute_error(labels, predictions)
  metrics['MAPE'] = skm.mean_absolute_percentage_error(labels, predictions)
  metrics['R2'] = skm.r2_score(labels, predictions)

  df_metrics = pd.DataFrame([metrics])

  return df_metrics