import json
import requests
import pandas as pd
import joblib
import numpy as np
from flask import jsonify

model_directory = 'model'
reg = f'{model_directory}/Model.pkl'
scaler=f'{model_directory}/scaler.pkl'

reg_final = joblib.load(reg)
print('model loaded')
scaler_final = joblib.load(scaler)
print('scaler loaded')
def predict():
	url='http://api.weatherapi.com/v1/forecast.json?key=385cf586f9c5428b85d201414211009&q={city}&days=3&aqi=no&alerts=no'.format(city="Atlanta")
	response=requests.get(url)
	response.raise_for_status()
	jsonResponse = response.json()
	idx=0
	Temperature= jsonResponse["forecast"]["forecastday"][idx]["day"]["maxtemp_f"]
	Dew_Point=jsonResponse["forecast"]["forecastday"][idx]["day"]["daily_will_it_snow"]
	Humidity=jsonResponse["forecast"]["forecastday"][idx]["day"]["avghumidity"]
	Wind_Speed=jsonResponse["forecast"]["forecastday"][idx]["day"]["maxwind_mph"]
	Pressure=jsonResponse["forecast"]["forecastday"][idx]["day"]["maxwind_kph"]
	Precipitation=jsonResponse["forecast"]["forecastday"][idx]["day"]["totalprecip_mm"]
	date="2021-12-24"
	time="12:35"
	month=int(date[-5:-3])
	hours=int(time[:-3])
	parameter = [Temperature,Dew_Point,Humidity,Wind_Speed,Pressure,Precipitation,month,hours]
	arr=np.array(parameter).reshape(1,-1)
	x_test = scaler_final.transform(arr)
	result=reg_final.predict(x_test)
	lists=result.tolist()
	return json.dumps(lists)
	
print(predict())

#json_2=	json.dumps(parameter)


#prediction = list(clf.predict(query))


#if __name__ == '__main__':

#    try:
#    	clf_final = joblib.load(clf)
#    	print('model loaded')
#    	scaler_final = joblib.load(scaler)
#    	print('scaler loaded')

#    except Exception as e:
#        print('No model here')
#        print('Train first')
#        print(str(e))
#        clf = None
