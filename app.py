from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get user input
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
    
    # Predict
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    
    return render_template("index.html", prediction_text=f"Result: {result}")

if __name__ == "__main__":
    app.run(debug=True)