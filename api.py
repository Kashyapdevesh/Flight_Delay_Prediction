import sys
import os
import shutil
import time
import traceback
import flask

from flask import Flask, request, jsonify
import pandas as pd
import joblib


app = Flask(__name__)

DEBUG = os.environ.get('DEBUG', False)
MODEL_NAME = os.environ.get('MODEL_NAME', 'model.joblib')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'local')
SERVICE_START_TIMESTAMP = time.time()
# inputs
training_data = 'data/final.csv'
include = ['Temperature','Dew Point','Humidity','Wind Speed','Pressure','Precipitation','month']
dependent_variable = 'WEATHER_DELAY'

model_directory = 'model'
clf = f'{model_directory}/model.joblib'
model_columns = f'{model_directory}/model_columns.joblib'

@app.route('/')
def home():
    return "Hi, Welcome to Flight Delay Prediction API"

@app.route('/predict', methods=['POST']) # Create http://host:port/predict POST end point
def predict():
    if clf:
        try:
            json_ = request.json #capture the json from POST
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(clf.predict(query))

            return jsonify({'prediction': [int(x) for x in prediction]})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'



@app.route('/wipe', methods=['GET']) # Create http://host:port/wipe GET end point
def wipe():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception as e:
        print(str(e))
        return 'Could not remove and recreate the model directory'

@app.route('/health')
def health_check():
    return flask.Response("up", status=200)

@app.route('/service-info')
def service_info():
    info = {
        'version-template': __version__,
        'running-since': SERVICE_START_TIMESTAMP,
        'serving-model-file': MODEL_NAME,
        'debug': DEBUG
    }
    return info


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        clf_final = joblib.load(clf)
        print('model loaded')
        model_columns_final = joblib.load(model_columns)
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        clf = None

	#app.run(host='0.0.0.0', port=port, debug=False)
    app.run(
        debug=DEBUG,
        host=os.environ.get('HOST', 'localhost'),
        port=os.environ.get('PORT', '5002'))
