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
def prediction():
  if request.method == 'POST':
    n_steps = 5
    past=[]
    for i in range(1, n_steps + 1):
      month = "month0" + str(i)
      result = float(request.form.get(month))
      past.append(result)

    row = asarray(past).reshape((1, n_steps, 1))
    yhat = model.predict(row)
    return render_template('result.html', result=yhat)
