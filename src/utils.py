import os
import pandas as pd

def load_data(file_path):
    """
    Load dataset from a given file path.

    Parameters:
    - file_path (str): Path to the dataset file.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)

def save_results(results, file_path):
    """
    Save results to a specified file path.

    Parameters:
    - results (pd.DataFrame or pd.Series): Results to be saved.
    - file_path (str): Path where results will be saved.
    """
    results.to_csv(file_path, index=False)

# Example usage
# data = load_data("data/sample_data.csv")
# save_results(predictions, "reports/predictions.csv")
