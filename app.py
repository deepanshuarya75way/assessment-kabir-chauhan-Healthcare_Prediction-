from flask import Flask, request, render_template, send_from_directory
import pickle
import numpy as np
import os 
import csv 
import json 
import uuid 
from datetime import datetime

app = Flask(__name__)

feedback_file = os.path.join(app.root_path, "feedback.csv")

def save_feedback(prediction_id, disease, inputs, prediction, feedback):
    file_exists = os.path.exists(feedback_file)
    with open(feedback_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "prediction_id", "disease", "input_data", "prediction", "feedback", "timestamp"
            ])
            writer.writerow([
                prediction_id, disease, json.dumps(inputs), prediction, feedback, datetime.now().isoformat()
            ])


@app.route('/plots/<path:filename>')
def plots(filename):
    return send_from_directory(os.path.join(app.root_path, 'plots'), filename)

model_diabetes = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-Healthcare_Prediction-/models/diabetes_model.pkl', 'rb'))
scalar_diabetes = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-Healthcare_Prediction-/models/diabetes_scaler.pkl', 'rb'))

model_heart = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-Healthcare_Prediction-/models/heart_model.pkl', 'rb'))
scalar_heart = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-Healthcare_Prediction-/models/heart_scaler.pkl', 'rb'))

model_parkinsons = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-Healthcare_Prediction-/models/parkinsons_model.pkl', 'rb'))
scalar_parkinsons = pickle.load(open(r'/home/coder/workspace/assessment-kabir-chauhan-Healthcare_Prediction-/models/parkinsons_scaler.pkl', 'rb'))

# Routes 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diabetes', methods = ['GET', 'POST'])
def diabetes():
    result = None
    result_class = ""
    prediction_id = None
    input_data = []

    if request.method == 'POST':
        # features = [float(x) for x in request.form.values()]
        # final_input = scalar_diabetes.transform([features])
        # prediction = model_diabetes.predict(final_input)[0]

        prediction_id = str(uuid.uuid4())
        
        input_data = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]


        features = input_data
        final_input = scalar_diabetes.transform([features])
        prediction = model_diabetes.predict(final_input)[0]

        print("User input:", features)
        print("Model prediction:", prediction)
         
        if prediction == 1:
            result = "Diabetes Detected"
            result_class = "result-bad"
        else:
            result = "No Diabetes Detected"
            result_class = "result-good"
    
    cm_plot = r'healthcare_project\plots\diabetes_cm.png'
    fi_plot = r'healthcare_project\plots\diabetes_features.png'
    
    return render_template('diabetes.html', result = result, result_class = result_class, cm_plot = cm_plot, fi_plot = fi_plot, prediction_id=prediction_id, input_data=input_data)

@app.route('/heart', methods = ['GET', 'POST'])
def heart():
    result = None
    result_class = ""
    prediction_id = None
    input_data = []

    if request.method == 'POST':
        prediction_id = str(uuid.uuid4())
        
        input_data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        features = input_data
        final_inputer = scalar_heart.transform([features])
        prediction = model_heart.predict(final_inputer)[0]
        print("User input:", features)
        print("Model prediction:", prediction)
        
        if prediction == 1:
            result = "No Heart disease Detected"
            result_class = "result-good"
        else:
            result = "Heart Disease Detected"
            result_class = "result-bad"
        
    cm_plot = r'healthcare_project\plots\heartdisease_cm.png'
    fi_plot = r'healthcare_project\plots\heartdisease_features.png'

    return render_template('heart.html', result = result, result_class = result_class, cm_plot = cm_plot, fi_plot = fi_plot,  prediction_id=prediction_id, disease="heart", input_data=json.dumps(input_data), prediction=prediction if result is not None else "")

@app.route('/parkinson', methods = ['GET', 'POST'])
def parkinson():
    result = None
    result_class = ""

    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_input = scalar_parkinsons.transform([features])
        prediction = model_parkinsons.predict(final_input)[0]
        
        if prediction == 1:
            result = "Parkinson Detected"
            result_class = "result-bad"
        else:
            result = "No Parkinson Detected"
            result_class = "result-good"
    
    cm_plot = r'healthcare_project\plots\parkinsons_cm.png'
    fi_plot = r'healthcare_project\plots\parkinsons_feature.png'
    
    return render_template('parkinson.html', result = result, result_class = result_class, cm_plot = cm_plot, fi_plot = fi_plot)

@app.route('/feedback', methods=['POST'])
def feedback():
    prediction_id = request.form.get('prediction_id')
    disease = request.form.get('disease')
    input_data = request.form.get('input_data')
    prediction = request.form.get('prediction')
    user_feedback = request.form.get('feedback')
    if not all([prediction_id, disease, input_data, prediction, user_feedback]):
        return "Invalid feedback.", 400
    if user_feedback not in ["correct", "incorrect"]:
        return "Invalid feedback.", 400
    try: 
        parsed_input=json.loads(input_data)
    except json.JSONDecodeError:
        return "Invalid input data. ", 400
    save_feedback(prediction_id, disease, parsed_input, prediction, user_feedback)
    if disease == "heart":
        back_url = "/heart"
        back_text = "Back to heart prediction"
    elif disease == "diabetes":
        back_url = "/diabetes"
        back_text = "Back to Diabetes Prediction"
    else: 
        back_url="/"
        back_text="Back to home"
    return render_template(
        'result.html', 
        prediction_id=prediction_id,
        disease=disease,
        feedback=user_feedback
    )

if __name__ == "__main__":
    app.run(debug=True)
    

