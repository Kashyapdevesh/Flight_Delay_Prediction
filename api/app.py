import sys
import os
import shutil
import time
import traceback
import flask
import json
import numpy as np

from flask import Flask, request,jsonify
import pandas as pd
import joblib

import requests 
from requests.exceptions import HTTPError

from endpoints.sample_predict import sample_predict

app = Flask(__name__)
app.register_blueprint(sample_predict) 

DEBUG = os.environ.get('DEBUG', False)
MODEL_NAME = os.environ.get('MODEL_NAME', 'Model.pkl')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'local')
SERVICE_START_TIMESTAMP = time.time()
__version__ = '1.0.0'
# inputs
training_data = 'data/final.csv'
include = ['Temperature','Dew Point','Humidity','Wind Speed','Pressure','Precipitation','month']
dependent_variable = 'WEATHER_DELAY'

cwd=os.getcwd()
model_directory = cwd+'/endpoints/models'
reg = f'{model_directory}/Model.pkl'
scaler=f'{model_directory}/scaler.pkl'

@app.route('/')
def home():
    return "Hi, Welcome to Flight Delay Prediction API"

@app.route('/predict', methods=['POST']) # Create http://host:port/predict POST end point
def predict():
    if reg:
        try:
            json_ = request.json
            if json_:
            	if 'iataCode' in json_:
            		iataCode = json_[19:22]
            	if 'date' in json_:
            		date = json_[38:48] #IN YYYY-MM-DD FORMAT
            	if 'time' in json_:
            		time =json_[64:69] #IN HH:MM FORMAT
            	with open('./airports.json') as json_file:
            		data=json.load(json_file)
            	airport = data[iataCode]
            	airport_city=airport['city']
            	try:
            		url='http://api.weatherapi.com/v1/forecast.json?key=385cf586f9c5428b85d201414211009&q={city}&days=3&aqi=no&alerts=no'.format(city=airport_city)
            		response=requests.get(url)
            		response.raise_for_status()
            		jsonResponse = response.json()
            		d0=jsonResponse["forecast"]["forecastday"][0]["date"]
            		d1=jsonResponse["forecast"]["forecastday"][1]["date"]
            		d2=jsonResponse["forecast"]["forecastday"][2]["date"]
            		dd_d0 = d0[-2:]
            		dd_d1 = d1[-2:]
            		dd_d2 = d2[-2:]
            		idx=0
            		if date[-2:] == dd_d0: #date[-2:]
            			idx=0
            		elif date[-2:] in dd_d2:
            			idx=1
            		else:
            			idx=2
            		Temperature= jsonResponse["forecast"]["forecastday"][idx]["day"]["maxtemp_f"]
            		Dew_Point=jsonResponse["forecast"]["forecastday"][idx]["day"]["daily_will_it_snow"]
            		Humidity=jsonResponse["forecast"]["forecastday"][idx]["day"]["avghumidity"]
            		Wind_Speed=jsonResponse["forecast"]["forecastday"][idx]["day"]["maxwind_mph"]
            		Pressure=jsonResponse["forecast"]["forecastday"][idx]["day"]["maxwind_kph"]
            		Precipitation=jsonResponse["forecast"]["forecastday"][idx]["day"]["totalprecip_mm"]
            		month=int(str(date[-5:-3]))
            		hours=int(str(time[:-3]))
            		parameter = [Temperature,Dew_Point,Humidity,Wind_Speed,Pressure,Precipitation,month,hours]
            				
            	except HTTPError as http_err:
            		print(f'HTTP error occurred: {http_err}')
            	except Exception as err:
            		print(f'Other error occurred: {err}')
            	arr=np.array(parameter).reshape(1,-1)
            	x_test = scaler_final.transform(arr)
            	result=reg_final.predict(x_test)
            	lists=result.tolist()
            	return json.dumps(lists)

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
    	reg_final = joblib.load(reg)
    	print('model loaded')
    	scaler_final = joblib.load(scaler)
    	print('scaler loaded')

    except Exception as e:
        print('No model here')
        print("main app")
        print('Train first')
        print(str(e))
        

	#app.run(host='0.0.0.0', port=port, debug=False)
    app.run(host='0.0.0.0')
