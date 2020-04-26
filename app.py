import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import xgboost as xgb

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Load data
features = pd.read_csv('features.csv')
revenue = pd.read_csv('revenue.csv')
revenue.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    movie_names = [x for x in request.form.values()]
    movie_name = movie_names[0]

    # subset dataframes
    features_subset = features[features['title'] == movie_name]
    revenue_subset = revenue[revenue['index'] == int(features_subset['index'])]
    actual_revenue = revenue_subset['revenue'].values[0]

    # Drop string features
    features_subset.drop(['title', 'imdb_id', 'index', 'Unnamed: 0'], axis=1, inplace=True)

    # Predict
    predicted_revenue = model.predict(xgb.DMatrix(features_subset.values))[0]

    # Round
    # actual_revenue = round(actual_revenue, 2)
    # predicted_revenue = round(predicted_revenue, 2)

    return (render_template('index.html', actual_text='Actual Box Office Revenue (Millions): $ {}'.format(actual_revenue),
        prediction_text='Predicted Box Office Revenue (Millions): $ {}'.format(predicted_revenue)))

# @app.route('/results',methods=['POST'])
# def results():
#
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#
#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
