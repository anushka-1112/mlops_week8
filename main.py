import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback

# Load dataset
df = pd.read_csv('data/iris.csv')

# Separate features and target
X = df.drop('species', axis=1)
y = df['species']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)  # Converts species names to integers

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 classes in Iris
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_val, y_val),
    verbose=0,
    callbacks=[TqdmCallback(), CSVLogger("metrics.csv")]
)

# Save model weights
model.save_weights("iris_model.weights.h5")
# Save the entire model (architecture + weights + optimizer state)
model.save("iris_model.h5")

