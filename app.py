from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# load your trained model
model = joblib.load("model/diabetes_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['Pregnancies']),
        float(request.form['Glucose']),
        float(request.form['BloodPressure']),
        float(request.form['SkinThickness']),
        float(request.form['Insulin']),
        float(request.form['BMI']),
        float(request.form['DiabetesPedigreeFunction']),
        float(request.form['Age'])
    ]

    prediction = model.predict([features])[0]

    result = "POSITIVE (Diabetic)" if prediction == 1 else "NEGATIVE (Not Diabetic)"

    return render_template('predict.html', prediction=result)

if __name__ == "__main__":
    app.run()
