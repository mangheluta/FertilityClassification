from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

dataset = pd.read_csv("fertility_Diagnosis.txt", header = None)

#print(dataset.head(5))

X = dataset.iloc[:,0:9]
Y = dataset.iloc[:, -1].replace(['N', 'O'], [0, 1])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = Sequential()
model.add(Dense(15, input_dim = 9, activation = 'relu')) 
model.add(Dense(10, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dropout(.2)) #prevents overfitting
model.add(Dense(1, activation = 'sigmoid')) # sigmoid instead of relu for final probability between 0 and 1

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 300, batch_size = 5, validation_data=(x_test, y_test))

#prediction
#X_new = np.array([1, 0.58, 1, 9, 3, 9, 9.6, 1, 0.5])
#X_new = np.reshape(X_new, (1, 9))

Y_new = model.predict_classes(x_test)

print('The accuracy is:',accuracy_score(y_test,Y_new))

print('Confusion matrix: ')
print(confusion_matrix(Y_new, y_test))