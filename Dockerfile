FROM tensorflow/tensorflow
WORKDIR /deeplearn
EXPOSE 5005
RUN pip install --upgrade pip numpy pandas sklearn flask numpy
COPY . .
RUN python3 mlp_iris_multiple_class.py iris.csv > mlp_log.txt 2>&1
ENTRYPOINT FLASK_APP=app.py flask run --host=0.0.0.0 --port=5005
