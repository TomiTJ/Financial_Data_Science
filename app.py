from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('optimised_loan_success_model.pkl')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.get_json()

        # Default values for missing features
        # Default values for missing features
        defaults = {
            "Amount": 5000,
            "Term": 36,
            "EmploymentType": "Employed - full time",  
            "ALL_CountDefaultAccounts": 0,
            "ALL_MeanAccountAge": 0.0,
            "ALL_TimeSinceMostRecentDefault": 0,
            "ALL_WorstPaymentStatusActiveAccounts": 0
        }
        # Validate numeric inputs
        numeric_fields = ["Amount", "Term", "ALL_CountDefaultAccounts", "ALL_MeanAccountAge", "ALL_TimeSinceMostRecentDefault", "ALL_WorstPaymentStatusActiveAccounts"]
        for field in numeric_fields:
            if not isinstance(input_data[field], (int, float)):
                return jsonify({'error': f"Invalid value for {field}. Must be a number."})
                
        # Merge user input with defaults
        input_data = {**defaults, **input_data}

        # Preprocess 'EmploymentType' (convert to one-hot encoded features)
        employment_types = ['Employed - full time', 'Employed - part time', 'Self employed', 'Retired']
        for et in employment_types:
            input_data[f"EmploymentType_{et}"] = 1 if input_data['EmploymentType'] == et else 0
        del input_data['EmploymentType']  # Remove the original column

        # Convert to DataFrame in the correct feature order
        expected_features = model.feature_names_in_
        input_df = pd.DataFrame([input_data], columns=expected_features)

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
        return jsonify({'error': f"Missing feature: {str(e)}. Please include this in your input."})
    except ValueError as e:
        return jsonify({'error': f"Invalid input: {str(e)}"})
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"})