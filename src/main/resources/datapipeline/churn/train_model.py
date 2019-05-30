# Code adapted from https://medium.com/@pushkarmandot/build-your-first-deep-learning-neural-network-model-using-keras-in-python-a90b5864116d

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('train/Churn_Modelling_00.csv')

X = dataset.iloc[:, 3:13].values
print("Before conversion")
print(X[:10,:])
y = dataset.iloc[:, 13].values

# Country names are replaced by 0,1 and 2 while male and female are replaced by 0 and 1.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#Reorder cols properly
X_data_frame = pd.DataFrame.from_records(X)
cols = list(X_data_frame.columns.values)
reorder=[3,0,1,2,4,5,6,7,8,9,10,11]
cols_new=[cols[i] for i in reorder]
X_data_frame = X_data_frame[cols_new]
X = X_data_frame.iloc[:,:].values
print("After conversion")
print(X[:10,:])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)
print("After featurization")
print(X_train[:10,:])


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


#Initializing Neural Network
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 12))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# Compiling Neural Network
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting our model 
classifier.fit(X_train, y, batch_size = 10, nb_epoch = 5)


# Predicting the training set results
y_pred = classifier.predict(X_train)
y_pred = (y_pred > 0.5)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

print(cm)

classifier.save("churn.h5")
