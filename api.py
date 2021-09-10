import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib


app = Flask(__name__)

# inputs
training_data = 'data/final.csv'
include = ['Temperature','Dew Point','Humidity','Wind Speed','Pressure','Precipitation','month']
dependent_variable = 'WEATHER_DELAY'

model_directory = 'model'
clf = f'{model_directory}/model.joblib'
model_columns = f'{model_directory}/model_columns.joblib'


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


@app.route('/ready')
def readiness_check():
    if model.is_ready():
        return flask.Response("ready", status=200)
    else:
        return flask.Response("not ready", status=503)


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        clf = joblib.load(model_file_name)
        print('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        clf = None

    app.run(host='0.0.0.0', port=port, debug=False)

