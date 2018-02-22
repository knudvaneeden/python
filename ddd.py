#
# Example from: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("dddtrain.csv", delimiter=",")
# split into input (X) and output (Y) variables
X_train = dataset[:,0:7036]
y_train = dataset[:,7036]
dataset = numpy.loadtxt("dddtest.csv", delimiter=",")
# split into input (X) and output (Y) variables
X_test = dataset[:,0:7036]
y_test = dataset[:,7036]
# create model
model = Sequential()
model.add(Dense(12, input_dim=7036, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict( X_test )
# round predictions
rounded = [ x[0] for x in predictions]
rounded = numpy.multiply( rounded, 100 )
rounded = numpy.round( rounded, decimals=0, out=None )
numpy.set_printoptions( threshold=numpy.inf )
print( rounded )
