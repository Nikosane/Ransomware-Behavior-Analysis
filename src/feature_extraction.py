import pandas as pd

def extract_features(data):
    """
    Extract relevant features from the dataset.

    Parameters:
    - data (pd.DataFrame): Raw dataset containing file operation logs.

    Returns:
    - pd.DataFrame: Dataset with extracted features.
    """
    # Example: Counting unique file operations
    data['operation_count'] = data.groupby('file_name')['operation'].transform('count')
    data['unique_files'] = data['file_name'].nunique()
    
    # Drop unnecessary columns
    features = data.drop(['file_name', 'operation'], axis=1)
    return features

# Example usage
# df = pd.read_csv("data/sample_data.csv")
# features = extract_features(df)
