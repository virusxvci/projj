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
app.secret_key = "m4xpl0it"

# Initialize database
db = SQLAlchemy(app)

# Load trained AI model for disease prediction
model_path = "model/random_forest.joblib"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model, vectorizer = joblib.load(model_path)

# User session storage
userSession = {}
all_result = {}

# Message constants
class msgCons:
    WELCOME_GREET = ["Hello!", "Hi there!", "Welcome!", "Greetings!", "Hey!"]

# User Model for Authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

# Home Page Route
import os

@app.route("/")
def home():
    template_path = os.path.join(app.root_path, "templates", "index.html")
    if not os.path.exists(template_path):
        return "Template Not Found: " + template_path, 404
    return render_template("index.html")


# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]
        user = User.query.filter_by(username=uname, password=passw).first()
        if user:
            session["user"] = uname  # Store user in session
            return redirect(url_for("home"))
    return render_template("login.html")

# Register Route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']
        new_user = User(username=uname, email=mail, password=passw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

# Logout Route
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

# Dashboard Route
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user_name=session["user"], appointments=[])

# Book Appointment Route
@app.route("/book-appointment", methods=["GET", "POST"])
def book_appointment():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        doctor_id = request.form.get("doctor")
        date = request.form.get("date")
        time = request.form.get("time")
        print(f"Booking appointment with Doctor ID: {doctor_id}, Date: {date}, Time: {time}")
        return redirect(url_for("dashboard"))
    
    doctors = [
        {"id": 1, "name": "Dr. Smith", "specialization": "Cardiologist"},
        {"id": 2, "name": "Dr. Johnson", "specialization": "Dermatologist"},
        {"id": 3, "name": "Dr. Lee", "specialization": "Pediatrician"},
    ]
    return render_template("book-appointment.html", doctors=doctors)

# Disease Prediction Function
def predict_disease_from_symptom(symptoms):
    input_vector = vectorizer.transform([", ".join(symptoms)])
    predicted_disease = model.predict(input_vector)[0]
    return predicted_disease, predicted_disease

# Chatbot Route
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

    return jsonify({'status': 'OK', 'answer': response})

# Run Flask App
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)