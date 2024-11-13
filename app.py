from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv("D:/PROJECT-01/artifacts/B_H_P.csv")
pipe = pickle.load(open("D:/PROJECT-01/artifacts/banglore_home_price_model.pkl", "rb"))

@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    total_sqft = request.form.get('total_sqft')

    print(location, bhk, bath, total_sqft)
    Inp = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(Inp)[0] * 1e5

    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)