import random
from flask import jsonify
import secrets
from flask import Flask, render_template, flash, redirect, url_for, session, logging, request, session
from flask_sqlalchemy import SQLAlchemy
from collections.abc import Mapping

# Configure logging


ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = "m4xpl0it"


def make_token():
    """
    Creates a cryptographically-secure, URL-safe string
    """
    return secrets.token_urlsafe(16) 
 
class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    email = db.Column(db.String(120))
    password = db.Column(db.String(80))


@app.route("/")
def index():
    return render_template("index.html")


userSession = {}

@app.route("/user")
def index_auth():
    my_id = make_token()
    userSession[my_id] = -1
    return render_template("index_auth.html", sessionId=my_id)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", user_name="User Name", appointments=[])

@app.route("/video-consultation")
def video_consultation():
    return render_template("video-consultation.html")

@app.route("/profile")
def profile():
    user_data = {
        "name": "User Name",
        "email": "user@example.com",
        "phone": "123-456-7890"
    }
    return render_template("profile.html", user=user_data)

@app.route("/book-appointment", methods=["GET", "POST"])
def book_appointment():
    if request.method == "POST":
        # Process form data
        doctor_id = request.form.get("doctor")
        date = request.form.get("date")
        time = request.form.get("time")

        # Here, you can save the appointment to a database or perform other actions
        print(f"Booking appointment with Doctor ID: {doctor_id}, Date: {date}, Time: {time}")

        # Redirect to a confirmation page or dashboard
        return redirect(url_for("dashboard"))

    # For GET requests, render the booking form
    doctors = [
        {"id": 1, "name": "Dr. Smith", "specialization": "Cardiologist"},
        {"id": 2, "name": "Dr. Johnson", "specialization": "Dermatologist"},
        {"id": 3, "name": "Dr. Lee", "specialization": "Pediatrician"},
    ]
    return render_template("book-appointment.html", doctors=doctors)


@app.route("/instruct")
def instruct():
    return render_template("instructions.html")

@app.route("/upload")
def bmi():
    return render_template("bmi.html")

@app.route("/diseases")
def diseases():
    return render_template("diseases.html")


@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    return render_template('pred.html', pred=pred, f_name=f_name)



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]

        login = user.query.filter_by(username=uname, password=passw).first()
        if login is not None:
            return redirect(url_for("index_auth"))
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']

        register = user(username=uname, email=mail, password=passw)
        db.session.add(register)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template("register.html")


import msgConstant as msgCons
import re

all_result = {
    'name':'',
    'age':0,
    'gender':'',
    'symptoms':[]
}


# Import Dependencies
# import gradio as gr
import pandas as pd
import numpy as np
from joblib import load
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def predict_symptom(user_input, symptom_list):
    # Convert user input to lowercase and split into tokens
    user_input_tokens = user_input.lower().replace("_"," ").split()

    # Calculate cosine similarity between user input and each symptom
    similarity_scores = []
    for symptom in symptom_list:
        # Convert symptom to lowercase and split into tokens
        symptom_tokens = symptom.lower().replace("_"," ").split()

        # Create count vectors for user input and symptom
        count_vector = np.zeros((2, len(set(user_input_tokens + symptom_tokens))))
        for i, token in enumerate(set(user_input_tokens + symptom_tokens)):
            count_vector[0][i] = user_input_tokens.count(token)
            count_vector[1][i] = symptom_tokens.count(token)

        # Calculate cosine similarity between count vectors
        similarity = cosine_similarity(count_vector)[0][1]
        similarity_scores.append(similarity)

    # Return symptom with highest similarity score
    max_score_index = np.argmax(similarity_scores)
    return symptom_list[max_score_index]




import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset into a pandas dataframe
try:
    df = pd.read_excel("/workspaces/projj/dataset.xlsx")
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")

# Get all unique symptoms
symptoms = set()
for s in df['Symptoms']:
    for symptom in s.split(','):
        symptoms.add(symptom.strip())


from joblib import load
import os
def predict_disease_from_symptom(symptom_list):
    try:
        # Verify model path
        model_path = "model/random_forest.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load the pre-trained model
        clf = load(model_path)
        print("Model loaded successfully.")

        # Prepare input for prediction
        symptoms = {
            'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
            'shivering': 0, 'chills': 0, 'joint_pain': 0, 'stomach_pain': 0, 'acidity': 0, 'ulcers_on_tongue': 0,
            'muscle_wasting': 0, 'vomiting': 0, 'burning_micturition': 0, 'spotting_ urination': 0, 'fatigue': 0,
            'weight_gain': 0, 'anxiety': 0, 'cold_hands_and_feets': 0, 'mood_swings': 0, 'weight_loss': 0,
            'restlessness': 0, 'lethargy': 0, 'patches_in_throat': 0, 'irregular_sugar_level': 0, 'cough': 0,
            'high_fever': 0, 'sunken_eyes': 0, 'breathlessness': 0, 'sweating': 0, 'dehydration': 0,
            'indigestion': 0, 'headache': 0, 'yellowish_skin': 0, 'dark_urine': 0, 'nausea': 0, 'loss_of_appetite': 0,
            'pain_behind_the_eyes': 0, 'back_pain': 0, 'constipation': 0, 'abdominal_pain': 0, 'diarrhoea': 0, 'mild_fever': 0,
            'yellow_urine': 0, 'yellowing_of_eyes': 0, 'acute_liver_failure': 0, 'fluid_overload': 0, 'swelling_of_stomach': 0,
            'swelled_lymph_nodes': 0, 'malaise': 0, 'blurred_and_distorted_vision': 0, 'phlegm': 0, 'throat_irritation': 0,
            'redness_of_eyes': 0, 'sinus_pressure': 0, 'runny_nose': 0, 'congestion': 0, 'chest_pain': 0, 'weakness_in_limbs': 0,
            'fast_heart_rate': 0, 'pain_during_bowel_movements': 0, 'pain_in_anal_region': 0, 'bloody_stool': 0,
            'irritation_in_anus': 0, 'neck_pain': 0, 'dizziness': 0, 'cramps': 0, 'bruising': 0, 'obesity': 0, 'swollen_legs': 0,
            'swollen_blood_vessels': 0, 'puffy_face_and_eyes': 0, 'enlarged_thyroid': 0, 'brittle_nails': 0, 'swollen_extremeties': 0,
            'excessive_hunger': 0, 'extra_marital_contacts': 0, 'drying_and_tingling_lips': 0, 'slurred_speech': 0,
            'knee_pain': 0, 'hip_joint_pain': 0, 'muscle_weakness': 0, 'stiff_neck': 0, 'swelling_joints': 0, 'movement_stiffness': 0,
            'spinning_movements': 0, 'loss_of_balance': 0, 'unsteadiness': 0, 'weakness_of_one_body_side': 0, 'loss_of_smell': 0,
            'bladder_discomfort': 0, 'foul_smell_of urine': 0, 'continuous_feel_of_urine': 0, 'passage_of_gases': 0, 'internal_itching': 0,
            'toxic_look_(typhos)': 0, 'depression': 0, 'irritability': 0, 'muscle_pain': 0, 'altered_sensorium': 0,
            'red_spots_over_body': 0, 'belly_pain': 0, 'abnormal_menstruation': 0, 'dischromic _patches': 0, 'watering_from_eyes': 0,
            'increased_appetite': 0, 'polyuria': 0, 'family_history': 0, 'mucoid_sputum': 0, 'rusty_sputum': 0, 'lack_of_concentration': 0,
            'visual_disturbances': 0, 'receiving_blood_transfusion': 0, 'receiving_unsterile_injections': 0, 'coma': 0,
            'stomach_bleeding': 0, 'distention_of_abdomen': 0, 'history_of_alcohol_consumption': 0, 'fluid_overload.1': 0,
            'blood_in_sputum': 0, 'prominent_veins_on_calf': 0, 'palpitations': 0, 'painful_walking': 0, 'pus_filled_pimples': 0,
            'blackheads': 0, 'scurring': 0, 'skin_peeling': 0, 'silver_like_dusting': 0, 'small_dents_in_nails': 0, 'inflammatory_nails': 0,
            'blister': 0, 'red_sore_around_nose': 0, 'yellow_crust_ooze': 0
        }

        # Normalize the symptom list to match the keys in the symptoms dictionary
        # Normalize the symptom list to match the keys in the symptoms dictionary
        normalized_symptoms = [s.strip().lower().replace(" ", "_") for s in symptom_list]

        # Set value to 1 for corresponding symptoms
        for s in normalized_symptoms:
            if s in symptoms:
                symptoms[s] = 1
            else:
                print(f"Symptom '{s}' not found in the symptoms dictionary.")

        # Prepare input for prediction
        # Prepare input for prediction
        input_data = pd.DataFrame([symptoms])
        print("Input data for prediction:")
        print(input_data)

        # Make prediction
        prediction = clf.predict(input_data)
        print(f"Prediction: {prediction}")

        return prediction[0], prediction[0]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error occurred during prediction.", ""
    

    # Set value to 1 for corresponding symptoms
    
    for s in symptom_list:
        index = predict_symptom(s, list(symptoms.keys()))
        print('User Input: ',s," Index: ",index)
        symptoms[index] = 1
    
    # Put all data in a test dataset
    df_test = pd.DataFrame(columns=list(symptoms.keys()))
    df_test.loc[0] = np.array(list(symptoms.values()))
    print(df_test.head()) 
    # Load pre-trained model
    clf = load(str("model/random_forest.joblib"))
    result = clf.predict(df_test)

    disease_details = getDiseaseInfo(result[0])
    
    # Cleanup
    del df_test
    
    return f"<b>{result[0]}</b><br>{disease_details}",result[0]



import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Get all unique diseases
diseases = set(df['Disease'])

def get_symptoms(user_disease):
    # Vectorize diseases using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Disease'])
    user_X = vectorizer.transform([user_disease])

    # Compute cosine similarity between user disease and dataset diseases
    similarity_scores = cosine_similarity(X, user_X)

    # Find the most similar disease(s)
    max_score = similarity_scores.max()
    print(max_score)
    if max_score < 0.7:
        print("No matching diseases found")
        return False,"No matching diseases found"
    else:
        max_indices = similarity_scores.argmax(axis=0)
        symptoms = set()
        for i in max_indices:
            if similarity_scores[i] == max_score:
                symptoms.update(set(df.iloc[i]['Symptoms'].split(',')))
        # Output results

        print("The symptoms of", user_disease, "are:")
        for sym in symptoms:
            print(str(sym).capitalize())

        return True,symptoms

from duckduckgo_search import DDGS

all_result = {'symptoms': []}  # Initialize data storage
userSession = {}  # Initialize session storage

def getDiseaseInfo(keywords):
    with DDGS() as ddgs:
        results = ddgs.text(keywords, region='wt-wt', safesearch='Off', time='y')
        print(f"DEBUG: DDG Results: {results}")  # Debugging
        return results


def predict_disease_from_symptom(symptoms):
    # Mock implementation for testing (replace with your logic)
    if symptoms:
        return "Flu", "Viral"
    return None, None


@app.route('/ask', methods=['GET', 'POST'])
def chat_msg():
    user_message = request.args.get("message", "").strip().lower()
    sessionId = request.args.get("sessionId", None)

    if not sessionId or sessionId not in userSession:
        return jsonify({'status': 'ERROR', 'answer': ['Invalid session. Please restart the chatbot.']})

    response = []
    currentState = userSession.get(sessionId, -1)

    print(f"DEBUG: Current State: {currentState}, User Message: {user_message}")

    if user_message == "undefined" or not user_message:
        response.append("Welcome to the chatbot! What is your name?")
        userSession[sessionId] = 0
        return jsonify({'status': 'OK', 'answer': response})

    if currentState == -1:
        response.append(f"Hi {user_message}, To predict your disease based on symptoms, we need some information about you. Please provide accordingly.")
        all_result['name'] = user_message
        userSession[sessionId] = 0

    elif currentState == 0:
        if user_message.isdigit() and 0 < int(user_message) < 130:
            all_result['age'] = int(user_message)
            response.append("Thank you! What symptoms are you experiencing?")
            response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
            userSession[sessionId] = 1
        else:
            response.append("Please provide a valid age.")

    elif currentState == 1:
        symptoms = [s.strip() for s in user_message.split(",")]
        if symptoms:
            all_result['symptoms'].extend(symptoms)
            response.append("Got it. Would you like to check for the disease now?")
            response.append("1. Yes")
            response.append("2. Add more symptoms")
            userSession[sessionId] = 2
        else:
            response.append("Please enter at least one symptom.")

    elif currentState == 2:
        if '1' in user_message or 'yes' in user_message:
            try:
                disease, disease_type = predict_disease_from_symptom(all_result['symptoms'])
                if not disease or not disease_type:
                    response.append("We could not find a disease matching your symptoms. Please try again.")
                else:
                    response.append(f"<b>The following disease may be causing your discomfort:</b> {disease}")
                    response.append(f'<a href="https://www.google.com/search?q={disease_type} disease hospital near me" target="_blank">Search Nearby Hospitals</a>')
                userSession[sessionId] = -1  # Reset session
            except Exception as e:
                print(f"Error predicting disease: {e}")
                response.append("There was an issue predicting your disease. Please try again.")
        elif '2' in user_message or 'add' in user_message:
            response.append("Please provide additional symptoms.")
            userSession[sessionId] = 1
        else:
            response.append("Invalid choice. Please type '1' to check disease or '2' to add more symptoms.")

    else:
        response.append("Something went wrong. Please restart the chatbot.")
        userSession[sessionId] = -1

    print(f"DEBUG: Next State: {userSession.get(sessionId)}")
    return jsonify({'status': 'OK', 'answer': response})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False, port=3000)