import unittest
import json
from flask import Flask

from unittest import TestCase
from unittest.mock import patch

import requests


#Snippet for importing module from parent directory
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from endpoints.sample_predict import sample_predict

app = Flask(__name__)
app.register_blueprint(sample_predict)

dictionary={ 		#Change these values for different test results
"iataCode": 'ATL',
"date":"2021-09-13",
"time":"03-23"}



class TestAnalytics(TestCase):

    @patch('requests.post')
    def test_post(self, mock_post):
        info = json.dumps(dictionary, indent = 4)
        resp = requests.post("http://0.0.0.0:5000/predict", data=json.dumps(info), headers={'Content-Type': 'application/json'})
        mock_post.assert_called_with("http://0.0.0.0:5000/predict", data=json.dumps(info), headers={'Content-Type': 'application/json'})


TestAnalytics().test_post()

url="http://0.0.0.0:5000/predict"
json_input = json.dumps(dictionary, indent = 4) 
r=requests.post(url,json=json_input)

print("The expected flight delay for the given test data is approximately: "+ str(r.json()))


if __name__ == '__main__':
    unittest.main()
