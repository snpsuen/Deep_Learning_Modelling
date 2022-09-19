import tensorflow as tf
from tensorflow import keras
from numpy import asarray
from flask import Flask, render_template, request, url_for

app = Flask(__name__)
model = tf.keras.models.load_model("./mlp_iris_multiple_class_model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/mlp_iris_index')
def mlp_iris_home():
    return render_template('mlp_iris_index.html')

@app.route('/mlp_iris_prediction', methods=['GET', 'POST'])
def mlp_iris_prediction():
  if request.method == 'POST':
    characteristics=[]
    for feature in ["sepallength", "sepalwidth", "petallength", "petalwidth"]
      result = float(request.form.get(feature))
      characteristics.append(result)

    yhat = model.predict(row)
    return render_template('result.html', result=yhat)
