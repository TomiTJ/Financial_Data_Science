from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the optimised model
model = joblib.load('optimised_loan_success_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Loan Success Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.get_json()

        # Convert input into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure features are in the correct order
        expected_features = model.feature_names_in_
        input_df = input_df[expected_features]

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Return response
        response = {
            'prediction': int(prediction),
            'success_probability': round(probability, 2)
        }
        return jsonify(response)

    except KeyError as e:
        return jsonify({'error': f"Missing feature: {str(e)}"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)