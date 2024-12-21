import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    def __init__(self, model_path, input_size):
        self.model = RansomwareModel(input_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def detect(self, data):
        """
        Detect anomalies in the given dataset.

        Parameters:
        - data (pd.DataFrame): Dataset containing features.

        Returns:
        - pd.Series: Predicted probabilities for each sample.
        """
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data_tensor = torch.tensor(data, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.model(data_tensor)
        return predictions.flatten()

# Example usage
# detector = AnomalyDetector("models/trained_model.pth", input_size=10)
# anomalies = detector.detect(new_data)
