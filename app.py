from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('stack.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('modelpred.html')

@app.route('/predict', methods=['POST'])
def predict():

    input_values = [float(x) for x in request.form.values()]

    input_df = pd.DataFrame([input_values], columns=['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'input8', 'input9', 'input10'])

    prediction = model.predict(input_df)

    return render_template('results.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
