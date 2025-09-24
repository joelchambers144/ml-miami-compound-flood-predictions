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
import pandas as pd

def create_input_columns(df: pd.DataFrame, input_specifications: list[dict]):
    """
    Creates lagged input columns based on the list of input specifications defined in the experiment
    configuration json file.
    
    Parameters:
        df (pd.DataFrame): Contains column names listed in input_specifications and a DateTime index
        input_specifications (list of dicts): List of dictionaries containing input column names and lag hour range
    
    Returns:
        df (pd.DataFrame): Contains lagged input columns
    """
    # Create input/output arrays
    for input in input_specifications:
      df = create_lagged_columns(df, input['column'], input['lag_range'])

    # Drop any NaNs that were created
    df.dropna(inplace=True)

    return df


def create_kfolds_keras(df_data: pd.DataFrame, train_years: list, target_features: str | list[str]):
    """
    Splits a pandas DataFrame into k-fold cross-validation splits. In the case of keras,
    a list of training-testing folds in the forms of dictionaries is created.
    
    Parameters:
        df_data (pd.DataFrame): DataFrame with 'date' column of type datetime
        train_years (list): List of years to be used as a test year in the cross-validation splits
        target_features (str or list of str): List of features to be separated from the rest of the input columns and used as the target variable
    
    Returns:
        kfolds (list of dicts): List of dictionaries containing the k-fold cross-validation splits
    """
    
    df_train_full = df_data.copy()

    df_inputs = df_train_full.drop(target_features, axis=1)
    feature_columns = df_inputs.columns

    kfolds = []
    for year in train_years:
        validation_year = [year]

        df_valid, df_train = split_df_by_years(df_train_full, validation_year)

        fold_train_years = sorted(df_train.index.year.unique())

        X_valid, y_valid = get_xy(df_valid, feature_columns, target_features)
        X_train, y_train = get_xy(df_train, feature_columns, target_features)

        fold = {
            'train_years' : fold_train_years,
            'train_data'  : (X_train, y_train),
            'valid_years' : validation_year,
            'valid_data'  : (X_valid, y_valid)
        }

        kfolds.append(fold)

    return kfolds


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


def get_xy(df: pd.DataFrame, feature_columns, target_features):
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
    y = df[target_features].to_numpy().ravel()
    X = df[feature_columns].to_numpy()

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
    
    # Include upper bound of range by adding +1
    lags = range(lag_range[0], lag_range[1] + 1)

    # Create lag columns first before adding to the existing df to
    # gain better performance
    # NOTE:: For the shift itself, the sign of the lag is flipped.
    # Reason: Shift treats the lags differently compared to how we treat them.
    # For example: When we ask for a shift of -24, we want the values 24 hours in the past.
    # However, shift will make it such that it instead gives you the values 24 hours in the future.
    # So, by flipping the sign the user will receive their desired result.
    new_cols = {
        f'{target_column}_t{"" if lag < 0 else "+"}{lag}': df[target_column].shift(-lag)
        for lag in lags
        if lag != 0
    }

    # Add the new lagged columns to the existing df
    df_lagged = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df_lagged


def extract_time_index(col_name):
    """
    Extracts the lag hour from the column name based on the format '_t[+ || -]{lag hour}'.
    If no '_t' suffix, then lag hour is 0.

    Parameters:
        col_name (str): Name of the column to extract the lag hour from

    Returns:
        int: Returns int found in column name. If no '_t' returns 0
    """
    if '_t' in col_name:
        # Extract the number after 't'. If positive, remove '+' symbol
        return int(col_name.split('_t')[-1].replace('+', ''))
    return 0


def order_input_arrays(df, column_names):
    """
    Takes a list of column names and sorts the columns in the DF based on:
        - The order passed in the column list
        - Ascending order by lag hour

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_names (list): Column names in the desired order to be sorted by

    Returns:
        pd.DataFrame: A DataFrame with columns sorted by input order and lag hour (ascending).
    """
    sorted_columns = []
    for column_prefix in column_names:
        sorted_columns = sorted_columns + sorted([col for col in df.columns if col.startswith(column_prefix)], key=extract_time_index)

    df_ordered = df[sorted_columns]

    return df_ordered