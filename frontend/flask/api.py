from flask import Flask, request
import pandas as pd
import pickle
import json
import boto3
import tensorflow as tf

app = Flask(__name__)


# def load_model():
#     s3 = boto3.client('s3',
#                     aws_access_key_id='AWS_KEY',
#                     aws_secret_access_key='AWS_SECRET_ACCESS_KEY'
#                     )
#     with open('saved_model.pb', 'wb') as mod:
#         s3.download_fileobj('dsml-alpha-2-covid-radiography-classification', 'data/fullmodel/saved_model.pb', mod)
#         new_model = tf.keras.models.load_model(mod)
#     return new_model

# model = load_model()

@app.route('/', methods = ['GET'])
def view():
    return 'Hello World!'

@app.route('/predict', methods = ['POST'])
def predict():
    new_model = tf.keras.models.load_model('../../model/fullmodel')
    data = request.json
    pred = new_model.predict(data)
    # pred = model.predict(data)
    return json.dumps(pred.tolist())
    # data    = request.json
    # return 'This is our model result!'

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)