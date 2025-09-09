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
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

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
  metrics['MSE'] = mean_squared_error(labels, predictions)
  metrics['RMSE'] = root_mean_squared_error(labels, predictions)
  metrics['MAE'] = mean_absolute_error(labels, predictions)
  metrics['MEDAE'] = median_absolute_error(labels, predictions)
  metrics['R2'] = r2_score(labels, predictions)

  df_metrics = pd.DataFrame([metrics])

  return df_metrics