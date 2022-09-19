import tensorflow as tf
import pickle
from tensorflow import keras
from numpy import asarray, argmax
from flask import Flask, render_template, request, url_for

app = Flask(__name__)
model = tf.keras.models.load_model("./mlp_iris_multiple_class_model")
fd = open('./mlp_iris_encoder.pkl', 'rb')
le = pickle.load(fd) 
fd.close()

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
    for label in ["sepallength", "sepalwidth", "petallength", "petalwidth"]:
      feature = float(request.form.get(label))
      characteristics.append(feature)

    yhat = model.predict([characteristics])
    classhat = le.inverse_transform([argmax(yhat)])
    return render_template('mlp_iris_result.html', features=characteristics, predictprob=yhat, predictclass=classhat)
