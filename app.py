from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Define the path to the models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# Load the models and preprocessing tools
log_reg = joblib.load(os.path.join(models_dir, 'logistic_regression_model.pkl'))
rf_clf = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
imputer = joblib.load(os.path.join(models_dir, 'imputer.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting inputs from the form
    senior_citizen = float(request.form['seniorCitizen'])
    tenure = float(request.form['tenure'])
    monthly_charges = float(request.form['monthlyCharges'])
    total_charges = float(request.form['totalCharges'])
    contract = request.form['contract']
    paperless_billing = request.form['paperlessBilling']
    payment_method = request.form['paymentMethod']
    gender = request.form['gender']
    partner = request.form['partner']
    dependents = request.form['dependents']

    # Encoding the categorical features
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    contract_encoded = contract_mapping[contract]

    paperless_billing_mapping = {'Yes': 1, 'No': 0}
    paperless_billing_encoded = paperless_billing_mapping[paperless_billing]

    payment_method_mapping = {
        'Credit card (automatic)': 0,
        'Electronic check': 1,
        'Mailed check': 2
    }
    payment_method_encoded = payment_method_mapping[payment_method]

    gender_mapping = {'Male': 1, 'Female': 0}
    gender_encoded = gender_mapping[gender]

    partner_mapping = {'Yes': 1, 'No': 0}
    partner_encoded = partner_mapping[partner]

    dependents_mapping = {'Yes': 1, 'No': 0}
    dependents_encoded = dependents_mapping[dependents]

    # Combine features into an array for prediction
    features = np.array([
        [senior_citizen, tenure, monthly_charges, total_charges,
         contract_encoded, paperless_billing_encoded, payment_method_encoded,
         gender_encoded, partner_encoded, dependents_encoded]
    ])
    
    # Apply necessary preprocessing (e.g., imputation, scaling)
    input_data_imputed = imputer.transform(features)
    input_data_scaled = scaler.transform(input_data_imputed)
    
    # Make predictions
    churn_prob_log_reg = log_reg.predict_proba(input_data_scaled)[0][1]
    churn_prob_rf = rf_clf.predict_proba(input_data_imputed)[0][1]
    
    # Render the predictions on the HTML page
    return render_template('index.html', 
                           log_reg_prob=round(churn_prob_log_reg * 100, 2),
                           rf_prob=round(churn_prob_rf * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
