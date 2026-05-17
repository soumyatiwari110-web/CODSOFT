# ============================================================
# TITANIC SURVIVAL PREDICTION - FLASK APP
# Author: Soumya Tiwari
# Description: Web application backend with prediction logic
# ============================================================

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# ----------------------------
# INITIALIZE FLASK APP
# ----------------------------
app = Flask(__name__)
app.secret_key = 'titanic_soumya_2024'

# ----------------------------
# LOAD TRAINED MODEL
# Load once at startup — don't reload on every request
# ----------------------------
MODEL_PATH = os.path.join('models', 'model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")

# ----------------------------
# HELPER: PREPROCESS INPUT
# Must match exact same steps done during training
# ----------------------------
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    """
    Convert form input into model-ready numeric array.
    """
    # Encode Sex: male=1, female=0
    sex_encoded = 1 if sex == 'male' else 0

    # Encode Embarked: C=0, Q=1, S=2
    embarked_map = {'C': 0, 'Q': 1, 'S': 2}
    embarked_encoded = embarked_map.get(embarked, 2)

    # Feature Engineering (same as training)
    family_size = int(sibsp) + int(parch) + 1
    is_alone = 1 if family_size == 1 else 0

    age = float(age)
    if age <= 12:
        age_group = 0
    elif age <= 18:
        age_group = 1
    elif age <= 35:
        age_group = 2
    elif age <= 60:
        age_group = 3
    else:
        age_group = 4

    fare = float(fare)
    if fare <= 7.91:
        fare_cat = 0
    elif fare <= 14.45:
        fare_cat = 1
    elif fare <= 31.0:
        fare_cat = 2
    else:
        fare_cat = 3

    # Build feature array in same order as training
    features = np.array([[
        int(pclass), sex_encoded, age, int(sibsp), int(parch),
        fare, embarked_encoded, family_size, is_alone, age_group, fare_cat
    ]])

    return features

# ----------------------------
# ROUTES
# ----------------------------

@app.route('/')
def home():
    """Home page — landing page with project intro"""
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    """Prediction form page"""
    return render_template('predict.html')


@app.route('/about')
def about():
    """About the project page"""
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles form submission.
    Reads passenger details → preprocesses → predicts → shows result.
    """
    try:
        # Get form data
        pclass   = request.form.get('pclass', 3)
        sex      = request.form.get('sex', 'male')
        age      = request.form.get('age', 30)
        sibsp    = request.form.get('sibsp', 0)
        parch    = request.form.get('parch', 0)
        fare     = request.form.get('fare', 15)
        embarked = request.form.get('embarked', 'S')

        # Preprocess input
        features = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Prepare result data
        survived = bool(prediction == 1)
        survival_prob = round(float(probability[1]) * 100, 1)
        death_prob = round(float(probability[0]) * 100, 1)

        # Passenger profile for display
        profile = {
            'pclass': {1: '1st Class (Upper)', 2: '2nd Class (Middle)', 3: '3rd Class (Lower)'}[int(pclass)],
            'sex': sex.capitalize(),
            'age': age,
            'sibsp': sibsp,
            'parch': parch,
            'fare': f'£{float(fare):.2f}',
            'embarked': {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}[embarked],
            'family_size': int(sibsp) + int(parch) + 1
        }

        return render_template('result.html',
                               survived=survived,
                               survival_prob=survival_prob,
                               death_prob=death_prob,
                               profile=profile)

    except Exception as e:
        return render_template('result.html',
                               error=str(e),
                               survived=False,
                               survival_prob=0,
                               death_prob=100,
                               profile={})
    
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')    


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for predictions"""
    try:
        data = request.get_json()
        features = preprocess_input(
            data['pclass'], data['sex'], data['age'],
            data['sibsp'], data['parch'], data['fare'], data['embarked']
        )
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        return jsonify({
            'survived': bool(prediction == 1),
            'survival_probability': round(float(probability[1]) * 100, 1),
            'death_probability': round(float(probability[0]) * 100, 1),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400


# ----------------------------
# RUN APP
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)