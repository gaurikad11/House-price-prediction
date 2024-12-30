from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle
import json
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Flask app and configure PostgreSQL URI
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define your Prediction model
class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    input_data = db.Column(db.String(200), nullable=False)
    predicted_value = db.Column(db.Float, nullable=False)

# Load the machine learning model and columns from pickle and JSON files
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Extract locations from the data columns (assuming locations start from index 3)
locations = data_columns[3:]

@app.route('/')
def home():
    # Pass the locations to the template
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bhk = int(request.form['bhk'])
    bathrooms = int(request.form['bathrooms'])
    location = request.form['location'].lower()

    try:
        loc_index = data_columns.index(location)
    except ValueError:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = area
    x[1] = bhk
    x[2] = bathrooms
    if loc_index >= 0:
        x[loc_index] = 1

    # Predict the price using the loaded model
    predicted_price = model.predict([x])[0]

    # Convert np.float64 to standard Python float
    predicted_price = float(predicted_price)

    # Save the prediction to PostgreSQL database
    new_prediction = Prediction(input_data=f'Area: {area}, BHK: {bhk}, Bathrooms: {bathrooms}, Location: {location}', predicted_value=predicted_price)
    db.session.add(new_prediction)
    db.session.commit()

    return render_template('index.html', locations=locations, predicted_price=round(predicted_price, 2))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # This creates the tables in the PostgreSQL database
    app.run(debug=True)
