import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv("Power_Demand_Data.csv")

# Feature engineering
features = data[["temperature", "time_of_day", "weekday"]]
target = data["demand"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["temperature", "time_of_day"]),
        ("cat", OneHotEncoder(), ["weekday"]),
    ]
)
features = preprocessor.fit_transform(features)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
