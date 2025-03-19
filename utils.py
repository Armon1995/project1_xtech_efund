"""
Created on Sat Mar 1 14:30:59 2024

Author: davideliu

E-mail: davide97ls@gmail.com

Goal: Utils functions
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pandas as pd
from tabulate import tabulate
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule


def plot_real_vs_predicted(real_value_series, pred_series):
    """
    Plots the real values and predicted values over time.

    Args:
        real_value_series (pandas.Series): The series of real values.
        pred_series (pandas.Series): The series of predicted values.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))

    # Plot the real values
    plt.plot(real_value_series.index, real_value_series, color='blue', label='Real Values', linewidth=2)

    # Plot the predicted values
    plt.plot(pred_series.index, pred_series, color='orange', label='Predicted Values', linestyle='--', linewidth=2)

    # Adding titles and labels
    plt.title('Real Values vs Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()  # Adjust layout to prevent overlap

    # Show the plot
    plt.show()


def compute_directional_accuracy(pred, actual, only_sign=False):
    """
    Computes the directional accuracy between predicted and actual values.

    Args:
        pred (numpy.ndarray): The predicted values.
        actual (numpy.ndarray): The actual values.
        only_sign (bool, optional): If True, compares only the signs of the values. Defaults to False.

    Returns:
        float: The directional accuracy.
    """
    actual = actual.values if not isinstance(actual, np.ndarray) else actual
    pred = pred.values if not isinstance(pred, np.ndarray) else pred
    last_actual = actual[:-1]
    diff_pred = pred[1:] - last_actual
    diff_actual = actual[1:] - last_actual
    if only_sign:
        return np.mean(np.sign(pred) == np.sign(actual))
    else:
        return np.mean(np.sign(diff_pred) == np.sign(diff_actual))


def compute_metrics(pred, actual):
    """
    Computes various metrics to evaluate the performance of predictions.

    Args:
        pred (numpy.ndarray or pandas.Series): The predicted values.
        actual (numpy.ndarray or pandas.Series): The actual values.

    Returns:
        tuple: A tuple containing the following metrics:
            - mse (float): Mean Squared Error
            - mae (float): Mean Absolute Error
            - mape (float): Mean Absolute Percentage Error
            - directional_accuracy (float): Directional Accuracy
            - r2 (float): R-squared
    """
    # Calculate directional accuracy
    directional_accuracy = compute_directional_accuracy(pred, actual)

    # Compute MSE, MAE, MAPE, and R-squared
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100  # Convert to percentage
    r2 = r2_score(actual, pred)

    return mse, mae, mape, directional_accuracy, r2
    

def print_metrics(results_dict):
    """
    Prints metrics for each dataset and their averages in a tabular format.

    Args:
        results_dict (dict): Dictionary containing dataset names as keys and tuples of (predictions, actual values) as values.
    """
    # Prepare data for tabulation in a single loop
    table_data = []
    headers = ['Dataset', 'MSE', 'MAE', 'MAPE (%)', 'Directional Accuracy', 'R-squared']

    # Initialize lists to store metric values for averaging
    mse_values = []
    mae_values = []
    mape_values = []
    directional_accuracy_values = []
    r2_values = []

    for dataset_name, (preds, reals) in results_dict.items():
        # Compute metrics
        mse, mae, mape, directional_accuracy, r2 = compute_metrics(preds, reals)
        
        # Append metrics to the table data
        table_data.append([
            dataset_name,
            f"{mse:.4f}",
            f"{mae:.4f}",
            f"{mape:.4f}",
            f"{directional_accuracy:.4f}",
            f"{r2:.4f}"
        ])
        
        # Store metrics for averaging
        mse_values.append(mse)
        mae_values.append(mae)
        mape_values.append(mape)
        directional_accuracy_values.append(directional_accuracy)
        r2_values.append(r2)

    # Compute averages for each metric
    avg_mse = np.mean(mse_values)
    avg_mae = np.mean(mae_values)
    avg_mape = np.mean(mape_values)
    avg_directional_accuracy = np.mean(directional_accuracy_values)
    avg_r2 = np.mean(r2_values)

    # Append average metrics to the table data
    table_data.append([
        "Average",
        f"{avg_mse:.4f}",
        f"{avg_mae:.4f}",
        f"{avg_mape:.4f}",
        f"{avg_directional_accuracy:.4f}",
        f"{avg_r2:.4f}"
    ])

    # Print the metrics in tabular format
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


def load_datasets_from_folder(folder_path):
    """
    Loads datasets from a specified folder.

    Args:
        folder_path (str): The path to the folder containing the datasets.

    Returns:
        dict: A dictionary containing dataset names as keys and pandas DataFrames as values.
    """
    datasets = {}
    
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv') or filename.endswith('.xlsx'):
            # Create the full file path
            file_path = os.path.join(folder_path, filename)
            # Load the dataset
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            else:
                df = pd.read_excel(file_path, index_col=0, parse_dates=True)
            # Use the filename without extension as the key
            dataset_name = os.path.splitext(filename)[0]
            datasets[dataset_name] = df
            
    return datasets


def prepare_ds_dataset(df, covariates, target_col, offset, prediction_length, windows, dist):
    """
    Prepares a dataset for use with GluonTS.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        covariates (list): A list of column names for the covariates.
        target_col (str): The name of the target column.
        offset (int): The offset for splitting the dataset.
        prediction_length (int): The length of the prediction horizon.
        windows (int): The number of windows for rolling window evaluation.
        dist (int): The distance between each window.

    Returns:
        tuple: A tuple containing the following:
            - train (PandasDataset): The training dataset.
            - test_data (list): A list of test data instances.
            - ds (PandasDataset): The original dataset.
    """
    # Create the PandasDataset
    ds = PandasDataset(
        df,
        feat_dynamic_real=covariates,
        target=target_col
    )

    # Split the dataset
    train, test_template = split(ds, offset=offset)

    # Generate test data instances
    test_data = test_template.generate_instances(
        prediction_length=prediction_length,  # number of time steps for each prediction
        windows=windows,  # number of windows in rolling window evaluation
        # number of time steps between each window - distance=prediction_length for non-overlapping windows
        distance=dist,
    )
    return train, test_data, ds


def generate_model(model_path, size, pdt, ctx, psz, fdrm, device, num_samples=100):
    if 'moe' in model_path:
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(model_path),
            prediction_length=pdt,
            context_length=ctx,
            patch_size=16,  # fixed patch size of 16
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=fdrm,
            past_feat_dynamic_real_dim=0,
        ).to(device)
    else:
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(model_path),
            prediction_length=pdt,
            context_length=ctx,
            patch_size=psz,
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=fdrm,
            past_feat_dynamic_real_dim=0,
        ).to(device)    
    return model