import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_excel("dataset.xlsx")  # Ensure this file exists

# Preprocess data
X = df["Symptoms"]  # Features (Symptom text)
y = df["Disease"]   # Labels (Disease name)

# Convert symptoms into numerical format
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save updated model
joblib.dump((model, vectorizer), "model/random_forest.joblib")
print("✅ Model retrained and saved successfully!")
