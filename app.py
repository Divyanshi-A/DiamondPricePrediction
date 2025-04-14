from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the trained model
if not os.path.exists('diamond_price_model.keras'):
    raise FileNotFoundError("Model file 'diamond_price_model.keras' not found. Please train the model first.")

model = tf.keras.models.load_model('diamond_price_model.keras')

# Load the preprocessor
if not os.path.exists('preprocessor.pkl'):
    raise FileNotFoundError("Preprocessor file 'preprocessor.pkl' not found. Please train the model first.")

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'carat': float(request.form['carat']),
            'cut': request.form['cut'],
            'color': request.form['color'],
            'clarity': request.form['clarity'],
            'depth': float(request.form['depth']),
            'table': float(request.form['table']),
            'x': float(request.form['x']),
            'y': float(request.form['y']),
            'z': float(request.form['z'])
        }
        
        input_df = pd.DataFrame([data])
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0][0]
        
        return render_template('index.html', prediction=f"Predicted Price: ${prediction:.2f}")
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        req_data = request.get_json()
        input_df = pd.DataFrame([req_data])
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0][0]
        
        return jsonify({"predicted_price": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

if __name__ == '__main__':
    app.run(debug=True)
