import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data
data_path = "../data/sample_data.csv"
df = pd.read_csv(data_path)

# Inspect the data
print("Initial Data Sample:")
print(df.head())

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Encode categorical columns
encoder = LabelEncoder()
df['operation'] = encoder.fit_transform(df['operation'])

# Normalize numerical features
scaler = StandardScaler()
df[['operation', 'label']] = scaler.fit_transform(df[['operation', 'label']])

# Save preprocessed data
df.to_csv("../data/preprocessed_data.csv", index=False)
print("Data preprocessing completed and saved.")

