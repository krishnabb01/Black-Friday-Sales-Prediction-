from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Label Encoders for categorical values (ensure consistency)
gender_encoder = LabelEncoder()
gender_encoder.fit(["F", "M"])

age_encoder = LabelEncoder()
age_encoder.fit(["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"])

city_encoder = LabelEncoder()
city_encoder.fit(["A", "B", "C"])

# Standard Scaler (Assuming we trained with it)
scaler = StandardScaler()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        gender = request.form["gender"]
        age = request.form["age"]
        occupation = int(request.form["occupation"])
        city_category = request.form["city_category"]
        stay_in_current_city_years = int(request.form["stay_in_current_city_years"])
        marital_status = int(request.form["marital_status"])
        product_category_1 = int(request.form["product_category_1"])
        product_category_2 = int(request.form["product_category_2"])
        product_category_3 = int(request.form["product_category_3"])

        # Encode categorical values
        gender = gender_encoder.transform([gender])[0]
        age = age_encoder.transform([age])[0]
        city_category = city_encoder.transform([city_category])[0]

        # Create feature array
        features = np.array([[gender, age, occupation, city_category,
                              stay_in_current_city_years, marital_status,
                              product_category_1, product_category_2, product_category_3]])

        # Apply standard scaling
        features_scaled = scaler.fit_transform(features)

        # Predict purchase amount
        prediction = model.predict(features_scaled)[0]

        return render_template("index.html", prediction_text=f"Estimated Purchase Amount: ₹{prediction:.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text="Error in processing input. Check values.")

if __name__ == "__main__":
    app.run(debug=True)
