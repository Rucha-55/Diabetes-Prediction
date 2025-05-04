import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load data
df = pd.read_csv("diabetes.csv")

# Preprocess: Replace zeros with median for relevant columns
columns_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_to_process:
    df[column] = df[column].replace(0, df[column].median())

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model with fixed random state for reproducibility
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save model with protocol=4 for better compatibility
joblib.dump(model, "model.pkl", protocol=4)