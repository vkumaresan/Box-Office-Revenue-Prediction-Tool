import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import xgboost as xgb
import gcsfs
import math
import os

pd.options.display.float_format = '{:,}'.format

# Load data and model
fs = gcsfs.GCSFileSystem(project='box-office-revenue-prediction')
with fs.open('box-office-revenue-prediction/features_train.csv') as f:
    features_train = pd.read_csv(f)
with fs.open('box-office-revenue-prediction/features_test.csv') as f:
    features_test = pd.read_csv(f)
with fs.open('box-office-revenue-prediction/revenue.csv') as f:
    revenue = pd.read_csv(f)

# features_train = pd.read_csv('../features_train.csv')
# features_test = pd.read_csv('../features_test.csv')
# revenue = pd.read_csv('../revenue.csv')

model = pickle.load(open('model.pkl', 'rb'))
revenue.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

TRAIN_FOLDER = 'train/'
TEST_FOLDER = 'test/'

app = Flask(__name__,static_url_path='')

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

        # poster
        path = str(str(features_subset['id'].values[0]) + '.jpeg')
        poster = os.path.join(TRAIN_FOLDER, path)

        # Budget
        budget = "${:,}".format(features_subset['budget'].values[0])

        # Release Date
        release_date =features_subset['release_date'].values[0]

        # Runtime
        runtime = features_subset['runtime'].values[0]

        # Tagline
        tagline =features_subset['tagline'].values[0]

        # Drop string features
        features_subset.drop(['title', 'imdb_id', 'index', 'Unnamed: 0', 'id', 'tagline', 'release_date', 'Unnamed: 0.1'], axis=1, inplace=True)

        # Predict
        predicted_revenue = math.exp(model.predict(xgb.DMatrix(features_subset.values))[0])

        # Round
        actual_revenue = round(actual_revenue)
        actual_revenue = "\n ${:,}".format(actual_revenue)
        predicted_revenue = round(predicted_revenue)
        predicted_revenue = "\n ${:,}".format(predicted_revenue)

    elif movie_name in movie_names_test:
        # subset dataframes
        features_subset = features_test[features_test['title'] == movie_name]
        revenue_subset = revenue[revenue['index'] == int(features_subset['index'])]
        actual_revenue = '\nNot found in dataset'

        # poster
        path = str(str(features_subset['id'].values[0]) + '.jpeg')
        poster = os.path.join(TEST_FOLDER, path)

        # Budget
        budget = "${:,}".format(features_subset['budget'].values[0])

        # Release Date
        release_date =features_subset['release_date'].values[0]

        # Runtime
        runtime = features_subset['runtime'].values[0]

        # Tagline
        tagline =features_subset['tagline'].values[0]

        # Drop string features
        features_subset.drop(['title', 'imdb_id', 'index', 'Unnamed: 0', 'id', 'tagline', 'release_date', 'Unnamed: 0.1'], axis=1, inplace=True)

        # Predict
        predicted_revenue = math.exp(model.predict(xgb.DMatrix(features_subset.values))[0])

        # Round
        #actual_revenue = round(actual_revenue, 2)
        predicted_revenue = round(predicted_revenue)
        predicted_revenue = "\n ${:,}".format(predicted_revenue)

    else:
        actual_revenue = ''
        predicted_revenue = ''
        poster = 'movie_theatre.jpeg'
        movie_name = '{} not found'.format(movie_name)
        budget=''
        release_date=''
        runtime=''
        tagline=''

    return (render_template('index.html',
        movie_text=movie_name,
        budget_text='Budget: {}'.format(budget),
        release_date_text='Release Date: {}'.format(release_date),
        runtime_text='Runtime (minutes): {}'.format(runtime),
        tagline_text=tagline,
        actual_text='Actual Box Office Revenue: {}'.format(actual_revenue),
        prediction_text='Predicted Box Office Revenue: {}'.format(predicted_revenue),
        user_image = poster))

@app.route('/results',methods=['POST'])
def results():
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

        # poster
        path = str(str(features_subset['id'].values[0]) + '.jpeg')
        poster = os.path.join(TRAIN_FOLDER, path)

        # Budget
        budget = "${:,}".format(features_subset['budget'].values[0])

        # Release Date
        release_date =features_subset['release_date'].values[0]

        # Runtime
        runtime = features_subset['runtime'].values[0]

        # Tagline
        tagline =features_subset['tagline'].values[0]

        # Drop string features
        features_subset.drop(['title', 'imdb_id', 'index', 'Unnamed: 0', 'id', 'tagline', 'release_date', 'Unnamed: 0.1'], axis=1, inplace=True)

        # Predict
        predicted_revenue = math.exp(model.predict(xgb.DMatrix(features_subset.values))[0])

        # Round
        actual_revenue = round(actual_revenue)
        actual_revenue = "${:,}".format(actual_revenue)
        predicted_revenue = round(predicted_revenue)
        predicted_revenue = "${:,}".format(predicted_revenue)

    elif movie_name in movie_names_test:
        # subset dataframes
        features_subset = features_test[features_test['title'] == movie_name]
        revenue_subset = revenue[revenue['index'] == int(features_subset['index'])]
        actual_revenue = 'Not found in dataset'

        # poster
        path = str(str(features_subset['id'].values[0]) + '.jpeg')
        poster = os.path.join(TEST_FOLDER, path)

        # Budget
        budget = "${:,}".format(features_subset['budget'].values[0])

        # Release Date
        release_date =features_subset['release_date'].values[0]

        # Runtime
        runtime = features_subset['runtime'].values[0]

        # Tagline
        tagline =features_subset['tagline'].values[0]

        # Drop string features
        features_subset.drop(['title', 'imdb_id', 'index', 'Unnamed: 0', 'id', 'tagline', 'release_date', 'Unnamed: 0.1'], axis=1, inplace=True)

        # Predict
        predicted_revenue = math.exp(model.predict(xgb.DMatrix(features_subset.values))[0])

        # Round
        #actual_revenue = round(actual_revenue, 2)
        predicted_revenue = round(predicted_revenue)
        predicted_revenue = "${:,}".format(predicted_revenue)

    else:
        actual_revenue = ''
        predicted_revenue = ''
        poster = 'movie_theatre.jpeg'
        movie_name = '{} not found'.format(movie_name)
        budget=''
        release_date=''
        runtime=''
        tagline=''

    output = { 'predicted_revenue': predicted_revenue, 'actual_revenue': actual_revenue }
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True,port=8080) # host='0.0.0.0'
    #app.run(debug=True)
