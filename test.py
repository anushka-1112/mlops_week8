import unittest
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

class TestModelAccuracy(unittest.TestCase):
    def setUp(self):
        # Load model
        self.model = load_model('iris_model.h5')

        # Load sample data
        df = pd.read_csv('sample.csv')
        self.X = df.drop('species', axis=1)
        y = df['species']

        # Encode labels
        self.le = LabelEncoder()
        self.le.fit(['setosa', 'versicolor', 'virginica'])
        self.y_encoded = self.le.transform(y)

        # Simulate the original training scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array([5.843, 3.054, 3.759, 1.199])
        self.scaler.scale_ = np.array([0.828, 0.433, 1.764, 0.763])
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = self.X.shape[1]

        self.X_scaled = self.scaler.transform(self.X)

    def test_accuracy_on_sample(self):
        _, accuracy = self.model.evaluate(self.X_scaled, self.y_encoded, verbose=0)
        print(f"\nModel accuracy on sample.csv: {accuracy:.2%}")
        self.assertGreater(accuracy, 0.95, "Model accuracy is not greater than 90% on sample.csv")

if __name__ == '__main__':
    unittest.main()
