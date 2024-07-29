import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model
logmodel = pickle.load(open('model0.pkl', 'rb'))

# Homepage route
@app.route('/')
def home():
    return render_template('home.html')

# Prediction API route
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Retrieve data from the POST request
        data = request.json['data']
        
        # Ensure the data is in the expected format
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid input data format'}), 400

        # Convert data to a numpy array
        input_data = np.array(list(data.values())).reshape(1, -1)
        
        # Make a prediction
        output = model0.predict(input_data)
        
        # Return prediction
        return jsonify({'prediction': int(output[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
