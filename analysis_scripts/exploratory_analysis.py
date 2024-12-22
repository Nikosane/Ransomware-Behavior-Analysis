import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_path = "../data/sample_data.csv"
df = pd.read_csv(data_path)

# Basic statistics
print("Dataset Statistics:")
print(df.describe())

# Timestamp distribution
plt.figure(figsize=(10, 6))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'].dt.hour.value_counts().sort_index().plot(kind='bar', title='Operations by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Operations')
plt.show()

# Operation type distribution
sns.countplot(data=df, x='operation')
plt.title('Distribution of Operation Types')
plt.show()

# Save plots for reports
plt.savefig("../reports/operation_distribution.png")
