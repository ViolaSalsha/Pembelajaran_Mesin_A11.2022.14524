import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# print(X)
# print(y)

# Encoding categori data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# labelencoder_X_1 = LabelEncoder()
# X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# labelencoder_X_2 = LabelEncoder()
# X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(),[1])], remainder = 'passthrough')
# X = onehotencoder.fit_transform(X)
# X = X[:, 2:]
# # print(X)

# splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

# Featuring scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# make the ANN
# importing the Keras Libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# initialising the ANN
classifier = Sequential()

# adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))


# Menambahkan layer kedua
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Menambahkan layer output
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

# fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)

# predicction the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# mkaing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Menghitung confusion matrix
cm = confusion_matrix(y_test, y_pred_)
print("Confusion Matrix:\n", cm)

# Menghitung metrik performa
accuracy = accuracy_score(y_test, y_pred_)
precision = precision_score(y_test, y_pred_)
recall = recall_score(y_test, y_pred_)
f1 = f1_score(y_test, y_pred_)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


