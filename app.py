from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "Pregnancies": float(request.form["pregnancies"]),
            "Glucose": float(request.form["glucose"]),
            "BloodPressure": float(request.form["bloodpressure"]),
            "SkinThickness": float(request.form["skinthickness"]),
            "Insulin": float(request.form["insulin"]),
            "BMI": float(request.form["bmi"]),
            "DiabetesPedigreeFunction": float(request.form["pedigree"]),
            "Age": float(request.form["age"]),
        }

        # Create DataFrame and handle any zero values
        df = pd.DataFrame([data])
        columns_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in columns_to_process:
            if df[column].iloc[0] == 0:
                return render_template("index.html", 
                    error=f"Please enter a valid non-zero value for {column}")

        # Scale the features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)[0]
        confidence = prediction_proba[1] if prediction[0] == 1 else prediction_proba[0]
        
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        return render_template("index.html", 
                             result=result, 
                             confidence=f"{confidence*100:.1f}%")
    
    except Exception as e:
        return render_template("index.html", 
                             error="Please enter valid numerical values for all fields")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)