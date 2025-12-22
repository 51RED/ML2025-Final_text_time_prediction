import torch
import numpy as np
from argparse import ArgumentParser, Namespace
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="regression",
        choices=["regression", "generative"],
    )
    return parser.parse_args()

def clear_memory():
    """Clear GPU memory safely"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    else:
        print("I don't know your device.")

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"

def evaluate_predictions(y_true, y_pred, set_name="Test"):
    """
    Calculate and print evaluation metrics.

    Args:
        y_true (np.array): True values (in minutes)
        y_pred (np.array): Predicted values (in minutes)
        set_name (str): Name of the dataset

    Returns:
        dict: Metrics dictionary
    """
    # Ensure non-negative predictions
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

    print(f"\n{set_name} Set Metrics:")
    print(f"  MAE: {mae:.2f} minutes")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return metrics
