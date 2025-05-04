import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("diabetes.csv")

# Preprocess: Replace zeros
df["Glucose"] = df["Glucose"].replace(0, df["Glucose"].median())

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")