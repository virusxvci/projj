import random
import re
import joblib
import secrets
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = "m4xpl0it"

# Load trained AI model for disease prediction
model_path = "model/random_forest.joblib"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

disease_model = joblib.load(model_path)

# Session & user data storage
userSession = {}
all_result = {}

# Message constants
class msgCons:
    WELCOME_GREET = ["Hello!", "Hi there!", "Welcome!", "Greetings!", "Hey!"]

def predict_disease_from_symptom(symptoms):
    """
    Uses the trained AI model to predict disease based on input symptoms.
    """
    input_text = ", ".join(symptoms)  # Convert list to comma-separated string
    predicted_disease = disease_model.predict([input_text])[0]
    return predicted_disease, predicted_disease  # Returning same label for now

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat_msg", methods=["GET"])
def chat_msg():
    user_message = request.args.get("message", "").lower()
    session_id = request.args.get("sessionId", "")

    response = []
    user_state = userSession.get(session_id, "NEW_USER")

    print(f"DEBUG: Current State: {user_state}, User Message: {user_message}")

    if user_message == "undefined":
        rand_num = random.randint(0, 4)
        response.append(msgCons.WELCOME_GREET[rand_num])
        response.append("What is your name?")
        return jsonify({'status': 'OK', 'answer': response})

    if user_state == "NEW_USER":
        response.append(f"Hi {user_message}, to predict your disease based on symptoms, we need some information about you.")
        userSession[session_id] = "ASK_AGE"
        all_result['name'] = user_message

    elif user_state == "ASK_AGE":
        response.append(f"{all_result['name']}, what is your age?")
        userSession[session_id] = "WAITING_FOR_AGE"

    elif user_state == "WAITING_FOR_AGE":
        pattern = r'\d+'
        age_match = re.findall(pattern, user_message)
        if not age_match or float(age_match[0]) <= 0 or float(age_match[0]) >= 130:
            response.append("Invalid input. Please provide a valid age.")
        else:
            all_result['age'] = float(age_match[0])
            response.append(f"{all_result['name']}, choose an option:")
            response.append("1. Predict Disease")
            response.append("2. Check Disease Symptoms")
            userSession[session_id] = "CHOOSE_ACTION"

    elif user_state == "CHOOSE_ACTION":
        if '2' in user_message or 'check' in user_message:
            response.append(f"{all_result['name']}, what's the disease name?")
            userSession[session_id] = "ASK_DISEASE_NAME"
        else:
            response.append(f"{all_result['name']}, what symptoms are you experiencing?")
            response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
            userSession[session_id] = "COLLECT_SYMPTOMS"

    elif user_state == "COLLECT_SYMPTOMS":
        all_result['symptoms'] = all_result.get('symptoms', [])
        all_result['symptoms'].extend(user_message.split(","))
        response.append(f"{all_result['name']}, what other symptoms are you experiencing?")
        response.append("1. Check Disease")
        response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
        userSession[session_id] = "CONFIRM_SYMPTOMS"

    elif user_state == "CONFIRM_SYMPTOMS":
        if '1' in user_message or 'disease' in user_message:
            print("DEBUG: Calling AI model for disease prediction...")
            try:
                disease, disease_type = predict_disease_from_symptom(all_result['symptoms'])
                print(f"DEBUG: Predicted Disease: {disease}")
                response.append("<b>The following disease may be causing your discomfort:</b>")
                response.append(disease)
                response.append(f'<a href="https://www.google.com/search?q={disease_type} disease hospital near me" target="_blank">Search Nearby Hospitals</a>')
                userSession[session_id] = "RESULT_DISPLAYED"
            except Exception as e:
                response.append("Error predicting disease. Please try again.")

    return jsonify({'status': 'OK', 'answer': response})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
