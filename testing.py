exteriorTemp = []
currentTemp = 0
weights = []
targetTemp = 22
time = 0000
maxTemp = 24
minTemp = 20
gradient = 0

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
theatreData = pd.read_csv('theatre_data.csv')
webData = pd.read_csv('web_data.csv')

temperatureData = numpy.zeros(shape(48,2))

for i in range 48:
    theatreTempData[i] = theatreData.iloc[:,-1][i].values    
    
for j in range 48: 
    webTempData[j] = webData.iloc[:,-1][j].values

#Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
theatreNorm = sc.fit_transform(webTempData)
theatreNorm = sc.fit_transform(theatreTempData)


#importing keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier= Sequential()

#adding the input layer and the first hidden layer
classifier.add(Dense(48,init='uniform', activation='relu', input_dim=11))
classifier.add(Dense(1,init='uniform', activation='sigmoid', input_dim=11))

#compile the ann
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])


# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred= (y_pred>0.5)

new_prediction= classifier.predict(sc.transform(np.array([[0.0,0,600, 1, 40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier.add(Dense(6,init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(6,init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(1,init='uniform', activation='sigmoid', input_dim=11))
    classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
    return classifier
    
classifier= KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
accuracies= cross_val_score(estimator=classifier, X= X_train, y= y_train, cv=10, n_jobs=-1)

mean=accuracies.mean()
variance= accuracies.std()

# improve the ANN by feature tuning 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.grid_search import GridSearchCV
def build_classifier():
    classifier.add(Dense(6,init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(6,init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(1,init='uniform', activation='sigmoid', input_dim=11))
    classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
    return classifier
    
classifier= KerasClassifier(build_fn=build_classifier)
parameters=     { 'batch_size': [25,32],
                 'nb_epoch':[100,500],
                 'optimizer':['adam', 'rmsprop']}
grid_search= GridSearchCV(estimator=classifier, 
                          param_grid= parameters,
                         scoring= 'accuracy',
                          cv=10)
grid_search= grid_search.fit(X_train, y_train)
best_parameters= grid_search.best_params_
best_accuracy= grid_search.best_score_
