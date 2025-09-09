#preprocessing.py
#----------------------------------
# Created By : Beto Estrada
#----------------------------------
""" 
This script holds functions for preprocessing the datasets and preparing
them for machine learning (ML) training.
""" 
#---------------------------------------------
# 
#
import numpy as np
import pandas as pd

def create_input_columns(df: pd.DataFrame, input_specifications: list[dict]):
    """
    Creates lagged input columns based on the list of input specifications defined in the experiment
    configuration json file.
    
    Parameters:
        df (pd.DataFrame): Contains column names listed in input_specifications and a DateTime index
        input_specifications (list of dicts): List of dictionaries containing input column names and lag hour range
    
    Returns:
        partitions (list of dicts): List of dictionaries containing the k-fold cross-validation splits
    """
    # Create input/output arrays
    for input in input_specifications:
      df = create_lagged_columns(df, input['column'], input['lag_range'])

    # Drop any NaNs that were created
    df.dropna(inplace=True)

    return df

def create_partitions_keras(df_data: pd.DataFrame, train_years: list, target_features: str | list[str]):
    """
    Splits a pandas DataFrame into k-fold cross-validation splits. In the case of keras,
    a list of training-testing folds in the forms of dictionaries is created.
    
    Parameters:
        df_data (pd.DataFrame): DataFrame with 'date' column of type datetime
        train_years (list): List of years to be used as a test year in the cross-validation splits
        target_features (str or list of str): List of features to be separated from the rest of the input columns and used as the target variable
    
    Returns:
        partitions (list of dicts): List of dictionaries containing the k-fold cross-validation splits
    """
    
    df_train_full = df_data.copy()

    df_inputs = df_train_full.drop(target_features, axis=1)
    feature_columns = df_inputs.columns

    partitions = []
    for year in train_years:
        validation_year = [year]

        df_valid, df_train = split_df_by_years(df_train_full, validation_year)

        partition_train_years = sorted(df_train.index.year.unique())

        X_valid, y_valid = get_xy(df_valid, feature_columns, target_features)
        X_train, y_train = get_xy(df_train, feature_columns, target_features)

        partition = {
            'train_years' : partition_train_years,
            'train_data'  : (X_train, y_train),
            'valid_years' : validation_year,
            'valid_data'  : (X_valid, y_valid)
        }

        partitions.append(partition)

    return partitions


def get_train_test_split(df_data, experiment, test_years):
    """
    Splits a pandas DataFrame into training and testing (or validation) sets. 
    
    Parameters:
        df_data (pd.DataFrame): DataFrame with 'date' column of type datetime
        experiment (Experiment): Experiment object containing experiment configurations
        test_years (list): List of test years to use as a filter for DataFrame
    
    Returns:
        X_train (array): Numpy array containing training predictor samples
        y_train (array): Numpy array containing training targets
        X_test (array): Numpy array containing testing predictor samples
        y_test (array): Numpy array containing testing targets
    """
    df_test, df_train = split_df_by_years(df_data, test_years)

    df_inputs = df_train.drop(experiment.target_column, axis=1)
    feature_columns = df_inputs.columns

    X_train, y_train = get_xy(df_train, feature_columns, experiment.target_column)
    X_test, y_test = get_xy(df_test, feature_columns, experiment.target_column)

    return X_train, y_train, X_test, y_test


def split_df_by_years(df_data, years):
    """
    Splits a pandas DataFrame into two separate DataFrames based on a list of years passed.
    df_a contains data from the given years. df_b contains data not in the list of years.
    
    Parameters:
        df_data (pd.DataFrame): DataFrame with 'date' column of type datetime
        years (list): List of years to use as a filter for DataFrame
    
    Returns:
        df_a (pd.DataFrame): DataFrame with data ONLY from the list of years passed
        df_a (pd.DataFrame): DataFrame with data NOT from the list of years passed
    """
    df_a = df_data[df_data.index.year.isin(years)]
    df_b = df_data[~df_data.index.year.isin(years)]
    
    return df_a, df_b


def get_xy(df, feature_columns, target_features):
    """
    Extracts the X (predictors) and y (targets) from a pandas DataFrame
    based on a list of feature columns and target feature columns passed.
    
    Parameters:
        df (pd.DataFrame): DataFrame with DateTime index
        feature_columns (list): List of feature column names
        target_features (str or list of str): Target feature(s)
    
    Returns:
        X (array): Numpy array containing predictor samples in the same order as the list of feature_columns passed
        y (array): Numpy array containing targets for each predictor sample
    """
    y = np.array(df[target_features]).ravel()
    X = df[feature_columns]

    X = np.array(X)

    return X, y


def create_lagged_columns(df, target_column, lag_range):
    """
    Adds lagged columns for a target column over a specified range of lags.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

        target_column (str): The column to create lagged values for.

        lag_range (tuple): A tuple (start, finish) representing the range of lags.
                           Negative values are past lags, positive values are future.

    Returns:
        pd.DataFrame: A DataFrame with added lagged columns.
    """
    for lag in range(lag_range[0], lag_range[1] + 1):  # Include upper bound of range
        if lag != 0:  # Avoid creating a lag for 0 as it's the same as the original column

            # NOTE:: For the shift itself, the sign of the lag is flipped.
            # Reason: Shift treats the lags differently compared to how we treat them.
            # For example: When we ask for a shift of -24, we want the values 24 hours in the past.
            # However, shift will make it such that it instead gives you the values 24 hours in the future.
            # So, by flipping the sign the user will receive their desired result.
            df[f'{target_column}_t{"" if lag < 0 else "+"}{lag}'] = df[target_column].shift(-lag)

    return df


def extract_time_index(col_name):
    if '_t' in col_name:
        # Extract the number after 't'
        return int(col_name.split('_t')[-1].replace('+', ''))
    return 0


def order_input_arrays(df, column_prefixes):
  sorted_columns = []
  for column_prefix in column_prefixes:
    sorted_columns = sorted_columns + sorted([col for col in df.columns if col.startswith(column_prefix)], key=extract_time_index)

  df_ordered = df[sorted_columns]

  return df_ordered






# NOTE:: Functions below are left behind for reference. Will be deleted eventually.


def extract_input_output_arrays(df, job_specification):
  """
  Here a dictionary is created for the given dataset. This dictionary contains:
  - The input array
  - The output array
  - The ordered input column names
  - And the original datetime indices for plotting the interactive time series later

  Parameters:
        df (pd.DataFrame): DataFrame containing dataset.

        job_specification (dict): Dictionary containing the job specification

  Returns:
      dict: Dictionary containing the input array, output array, ordered input columns, and datetime index
  """
  input_specifications = job_specification['input_specifications']
  target_column = job_specification['target_column']

  for input in input_specifications:
      df = create_lagged_columns(df, input['column'], input['lag_range'])

  df.dropna(inplace=True)

  target_array = df[target_column].values

  # Save datetimes of df for plotting purposes later
  datetime_index = df.index

  # Remove the target column from df before creating ordered input arrays
  df_inputs = df.drop(columns=[target_column])

  column_prefixes = job_specification['column_prefixes']

  df_inputs_ordered = order_input_arrays(df_inputs, column_prefixes)

  # Ordered input column names saved for plotting purposes later
  ordered_input_list = df_inputs_ordered.columns.tolist()

  input_array = df_inputs_ordered.values

  input_output_dict = {'input_array': input_array,
                       'target_array': target_array,
                       'ordered_input_list': ordered_input_list,
                       'datetime_index': datetime_index}

  return input_output_dict


def extract_input_output_arrays2(df, experiment):
    """
    Here a dictionary is created for the given dataset. This dictionary contains:
    - The input array
    - The output array
    - The ordered input column names
    - And the original datetime indices for plotting the interactive time series later

    Parameters:
        df (pd.DataFrame): DataFrame containing dataset.

        job_specification (dict): Dictionary containing the job specification

    Returns:
        dict: Dictionary containing the input array, output array, ordered input columns, and datetime index
    """
    target_column = experiment['target_column']

    target_array = df[target_column].values

    # Remove the target column from df before creating ordered input arrays
    df_inputs = df.drop(columns=[target_column])

    column_prefixes = experiment['column_prefixes']

    df_inputs_ordered = order_input_arrays(df_inputs, column_prefixes)

    input_array = df_inputs_ordered.values

    return input_array, target_array