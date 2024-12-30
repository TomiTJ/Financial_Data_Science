from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for cross-origin requests
CORS(app)

# Load the trained model
model = joblib.load('optimised_loan_success_model.pkl')

# Default endpoint for home
@app.route('/')
def home():
    return render_template('form.html')  # Ensure form.html exists in the 'templates' folder

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check content type
        if request.content_type == 'application/json':
            input_data = request.get_json()
        else:
            # Parse form data
            input_data = {
                "Amount": request.form.get("Amount", type=int),
                "Term": request.form.get("Term", type=int),
                "EmploymentType": request.form.get("EmploymentType", type=str),
                "ALL_CountDefaultAccounts": request.form.get("ALL_CountDefaultAccounts", type=int, default=0),
                "ALL_MeanAccountAge": request.form.get("ALL_MeanAccountAge", type=float, default=0.0),
                "ALL_TimeSinceMostRecentDefault": request.form.get("ALL_TimeSinceMostRecentDefault", type=int, default=0),
                "ALL_WorstPaymentStatusActiveAccounts": request.form.get("ALL_WorstPaymentStatusActiveAccounts", type=int, default=0)
            }


        # Preprocess EmploymentType
        employment_types = ['Employed - full time', 'Employed - part time', 'Self employed', 'Retired']
        for et in employment_types:
            input_data[f"EmploymentType_{et}"] = 1 if input_data['EmploymentType'] == et else 0
        del input_data['EmploymentType']

        # Convert to DataFrame in the correct feature order
        expected_features = model.feature_names_in_
        input_df = pd.DataFrame([input_data], columns=expected_features)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Return the prediction as a response
        return jsonify({
            'prediction': int(prediction),
            'success_probability': round(probability, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

    except KeyError as e:
        return jsonify({'error': f"Missing feature: {str(e)}. Please include this in your input."})
    except ValueError as e:
        return jsonify({'error': f"Invalid input: {str(e)}"})
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)