#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Local version: scores using a locally trained scikit-learn model
# (model.pkl, produced by train_model.py) instead of calling out to an
# IBM Cloud Pak for Data deployment.

import os
import joblib
import pandas as pd
from flask import Flask, request, session, render_template, flash

app = Flask(__name__)

app.config.update(dict(
    DEBUG=True,
    SECRET_KEY=os.environ.get('SECRET_KEY', 'development key')
))

strings = {
    "gender": ['Female', 'Male'],
    "Partner": ['Yes', 'No'],
    "Dependents": ['No', 'Yes'],
    "PhoneService": ['No', 'Yes'],
    "MultipleLines": ['No phone service', 'No', 'Yes'],
    "InternetService": ['DSL', 'Fiber optic', 'No'],
    "OnlineSecurity": ['No', 'Yes', 'No internet service'],
    "OnlineBackup": ['Yes', 'No', 'No internet service'],
    "DeviceProtection": ['No', 'Yes', 'No internet service'],
    "TechSupport": ['No', 'Yes', 'No internet service'],
    "StreamingTV": ['No', 'Yes', 'No internet service'],
    "StreamingMovies": ['No', 'Yes', 'No internet service'],
    "Contract": ['Month-to-month', 'One year', 'Two year'],
    "PaperlessBilling": ['Yes', 'No'],
    "PaymentMethod": ['Electronic check',
                      'Mailed check',
                      'Bank transfer (automatic)',
                      'Credit card (automatic)']
}

# min, max, default value
floats = {
    "MonthlyCharges": [0, 1000, 100],
    "TotalCharges": [0, 50000, 1000]
}

# min, max, default value
ints = {
    "SeniorCitizen": [0, 1, 0],
    "tenure": [0, 100, 2],
}

labels = ["No Churn", "Churn"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
_bundle = None


def get_model_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f'model.pkl not found at {MODEL_PATH}. '
                f'Run "python train_model.py" first to create it.'
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def generate_input_lines():
    result = f'<table>'

    counter = 0
    for k in floats.keys():
        minn, maxx, vall = floats[k]
        if (counter % 2 == 0):
            result += f'<tr>'
        result += f'<td>{k}'
        result += f'<input type="number" class="form-control" min="{minn}" max="{maxx}" step="1" name="{k}" id="{k}" value="{vall}" required (this.value)">'
        result += f'</td>'
        if (counter % 2 == 1):
            result += f'</tr>'
        counter = counter + 1

    counter = 0
    for k in ints.keys():
        minn, maxx, vall = ints[k]
        if (counter % 2 == 0):
            result += f'<tr>'
        result += f'<td>{k}'
        result += f'<input type="number" class="form-control" min="{minn}" max="{maxx}" step="1" name="{k}" id="{k}" value="{vall}" required (this.value)">'
        result += f'</td>'
        if (counter % 2 == 1):
            result += f'</tr>'
        counter = counter + 1

    counter = 0
    for k in strings.keys():
        if (counter % 2 == 0):
            result += f'<tr>'
        result += f'<td>{k}'
        result += f'<select class="form-control" name="{k}">'
        for value in strings[k]:
            result += f'<option value="{value}" selected>{value}</option>'
        result += f'</select>'
        result += f'</td>'
        if (counter % 2 == 1):
            result += f'</tr>'
        counter = counter + 1

    result += f'</table>'

    return result


app.jinja_env.globals.update(generate_input_lines=generate_input_lines)


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        data = {}

        for k, v in request.form.items():
            data[k] = v
            session[k] = v

        for field in ints.keys():
            data[field] = int(data[field])
        for field in floats.keys():
            data[field] = float(data[field])

        bundle = get_model_bundle()
        model = bundle["model"]
        feature_order = bundle["features"]

        row = pd.DataFrame([{f: data[f] for f in feature_order}])
        proba = model.predict_proba(row)[0]  # [P(no churn), P(churn)]
        no_percent = float(proba[0]) * 100
        yes_percent = float(proba[1]) * 100
        churn_risk = "yes" if yes_percent >= 50 else "no"

        flash('Percentage of this customer leaving is: %.0f%%' % yes_percent)
        return render_template(
            'score.html',
            probability=[proba[0], proba[1]],
            churn_risk=churn_risk,
            yes_percent=yes_percent,
            no_percent=no_percent,
            labels=labels)

    else:
        return render_template('input.html')


port = os.environ.get('PORT', '5000')
host = os.environ.get('HOST', '0.0.0.0')
if __name__ == "__main__":
    app.run(host=host, port=int(port))
