from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

# Load trained models and scaler
lr_model = pickle.load(open("models/lr_model.pkl", "rb"))
dt_model = pickle.load(open("models/dt_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form submission
        form_data = request.form
        input_features = ['LotArea', 'OverallQual', 'YearBuilt', 'GrLivArea', 'FullBath', 'BedroomAbvGr']
        
        # Convert form data to array
        input_data = np.array([float(form_data[feature]) for feature in input_features]).reshape(1, -1)
        
        # Scale input features
        input_data = scaler.transform(input_data)
        
        # Make predictions
        lr_prediction = lr_model.predict(input_data)[0]
        dt_prediction = dt_model.predict(input_data)[0]

        # Format the predictions in INR (₹)
        return render_template('index.html', 
                               prediction_text=f"Predicted Price (Linear Regression): ₹{round(lr_prediction, 2)} | (Decision Tree): ₹{round(dt_prediction, 2)}")
    except Exception as e:
        return render_template('index.html', error_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
