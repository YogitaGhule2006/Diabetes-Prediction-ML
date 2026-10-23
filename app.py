
from pathlib import Path

from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved model bundle from the project directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model_bundle.pkl"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
scaler = bundle["scaler"]
model_features = bundle["features"]

gender_categories = bundle["gender_categories"]
smoking_categories = bundle["smoking_categories"]


@app.route("/")
def home():
    return render_template(
        "index.html",
        genders=gender_categories,
        smoking_options=smoking_categories
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect patient details from the web form
        gender = request.form["gender"]
        age = float(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        smoking_history = request.form["smoking_history"]
        bmi = float(request.form["bmi"])
        hba1c_level = float(request.form["hba1c_level"])
        blood_glucose_level = float(
            request.form["blood_glucose_level"]
        )

        # Validate numeric inputs
        values = [
            age,
            bmi,
            hba1c_level,
            blood_glucose_level
        ]

        if not np.isfinite(values).all():
            raise ValueError("Please enter valid finite numbers.")

        if not 0 <= age <= 120:
            raise ValueError("Age must be between 0 and 120.")

        if hypertension not in (0, 1):
            raise ValueError("Invalid hypertension value.")

        if heart_disease not in (0, 1):
            raise ValueError("Invalid heart disease value.")

        if bmi <= 0 or hba1c_level <= 0 or blood_glucose_level <= 0:
            raise ValueError(
                "BMI, HbA1c level, and blood glucose level must be positive."
            )

        if gender not in gender_categories:
            raise ValueError("Invalid gender selection.")

        if smoking_history not in smoking_categories:
            raise ValueError("Invalid smoking history selection.")

        # Create a DataFrame using the original column names
        user_data = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "hba1c_level": hba1c_level,
            "blood_glucose_level": blood_glucose_level
        }])

        # Encode categorical columns as done during training
        user_encoded = pd.get_dummies(
            user_data,
            columns=["gender", "smoking_history"],
            dtype=int
        )

        # Match training feature names and order
        user_encoded = user_encoded.reindex(
            columns=model_features,
            fill_value=0
        )

        # Apply the previously fitted scaler
        user_scaled = scaler.transform(user_encoded)

        user_input = pd.DataFrame(
            user_scaled,
            columns=model_features
        )

        # Predict using the trained model
        prediction = int(model.predict(user_input)[0])
        probabilities = model.predict_proba(user_input)[0]

        # Get the probability associated with class 1
        class_list = list(model.classes_)

        if 1 not in class_list:
            raise ValueError(
                "The loaded model does not contain the diabetes-positive class."
            )

        diabetes_index = class_list.index(1)
        diabetes_probability = float(
            probabilities[diabetes_index] * 100
        )

        if prediction == 1:
            result = "Diabetes Predicted"
        else:
            result = "Diabetes Not Predicted"

        return render_template(
            "result.html",
            result=result,
            prediction=prediction,
            probability=round(diabetes_probability, 2),
            patient={
                "gender": gender,
                "age": age,
                "hypertension": "Yes" if hypertension else "No",
                "heart_disease": "Yes" if heart_disease else "No",
                "smoking_history": smoking_history,
                "bmi": bmi,
                "hba1c_level": hba1c_level,
                "blood_glucose_level": blood_glucose_level
            }
        )

    except (ValueError, TypeError, KeyError) as error:
        return render_template(
            "error.html",
            message=str(error)
        ), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
    "error.html",
    message="Prediction failed. Check the server terminal for details."
), 500


if __name__ == "__main__":
    # Debug mode is for local development only
    app.run(debug=True)
