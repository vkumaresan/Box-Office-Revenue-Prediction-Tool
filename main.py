import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import xgboost as xgb
import gcsfs
import math

app = Flask(__name__)

# Load data and model
fs = gcsfs.GCSFileSystem(project='cloud-computing-272319')
with fs.open('cloud-computing-272319.appspot.com/features_train.csv') as f:
    features_train = pd.read_csv(f)
with fs.open('cloud-computing-272319.appspot.com/features_test.csv') as f:
    features_test = pd.read_csv(f)
with fs.open('cloud-computing-272319.appspot.com/revenue.csv') as f:
    revenue = pd.read_csv(f)

model = pickle.load(open('model.pkl', 'rb'))
# features_train = pd.read_csv('features_train.csv')
# features_test = pd.read_csv('features_test.csv')
# revenue = pd.read_csv('revenue.csv')
revenue.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    movie_names = [x for x in request.form.values()]
    movie_name = movie_names[0]

    # get list of unique movie names in train and test
    movie_names_train = list(features_train['title'].unique())
    movie_names_test = list(features_test['title'].unique())

    if movie_name in movie_names_train:
        # subset dataframes
        features_subset = features_train[features_train['title'] == movie_name]
        revenue_subset = revenue[revenue['index'] == int(features_subset['index'])]
        actual_revenue = math.exp(revenue_subset['revenue'].values[0])

        # Drop string features
        features_subset.drop(['title', 'imdb_id', 'index', 'Unnamed: 0'], axis=1, inplace=True)

        # Predict
        predicted_revenue = math.exp(model.predict(xgb.DMatrix(features_subset.values))[0])

        # Round
        actual_revenue = round(actual_revenue, 2)
        predicted_revenue = round(predicted_revenue, 2)

    elif movie_name in movie_names_test:
        # subset dataframes
        features_subset = features_test[features_test['title'] == movie_name]
        revenue_subset = revenue[revenue['index'] == int(features_subset['index'])]
        actual_revenue = 'Not found in dataset'

        # Drop string features
        features_subset.drop(['title', 'imdb_id', 'index', 'Unnamed: 0'], axis=1, inplace=True)

        # Predict
        predicted_revenue = math.exp(model.predict(xgb.DMatrix(features_subset.values))[0])

        # Round
        #actual_revenue = round(actual_revenue, 2)
        predicted_revenue = round(predicted_revenue, 2)

    else:
        actual_revenue = 'Movie data not found'
        predicted_revenue = 'Movie data not found'

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
    app.run(debug=True,port=8080) # host='0.0.0.0'
    #app.run(debug=True)
