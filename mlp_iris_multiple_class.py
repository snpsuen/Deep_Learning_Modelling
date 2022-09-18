# -*- coding: utf-8 -*-
"""mlp_iris_multiclass_classname.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/snpsuen/9b3d53b6aae9e0470015f0d00a1033c4/mlp_iris_multiclass_classname.ipynb
"""

# mlp for multiclass classification
import sys
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# load the dataset
inputfile = sys.argv[1]
path = f'./{inputfile}'
df = read_csv(path, header=None)

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# enumerate the classification strings for the output
le = preprocessing.LabelEncoder()
le.fit(y)
print("le.classes_=%s" % (le.classes_))
# encode the output strings to integers
y = le.transform(y)
# y = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print("X_train.shape=%s, X_test.shape=%s, y_train.shape=%s, y_test.shape=%s" %(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
# determine the number of input features
n_features = X_train.shape[1]
n_xpoints = X_train.shape[0]
n_ypoints = y_train.shape[0]

print('Number of features: %d' % (n_features))
print('Number of X data points: %d' % (n_xpoints))
print('Number of y data points: %d' % (n_ypoints))

# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(5, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

# make a prediction
row = [5.1,3.5,1.4,0.2]
yhat = model.predict([row])
predicted = le.classes_[argmax(yhat)]
print('Prediction for %s = %s (class = %s)' % (row, yhat, predicted))

# save the model
model.save("./mlp_iris_multiple_class_model")
print('Model saved as rnmlp_iris_multiple_class_model')
