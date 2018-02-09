
import csv

import keras

import numpy as np

from keras.datasets import mnist

np.random.seed(42)

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

print( "reading X_train" )
datafile = open( 'ddd11.csv', 'r' )
datareader = csv.reader( datafile, delimiter=',' )
X_train = []
for row in datareader:
 X_train.append( row )

print( "reading y_train" )
y_train = []
dataReader = csv.reader( open( 'ddd12.csv' ), delimiter=',', quotechar='"' ) # csv file has to be in the same directory as this Python .py file
for row in dataReader:
 y_train.append( row )

print( "reading X_test" )
datafile = open( 'ddd21.csv', 'r' )
datareader = csv.reader( datafile, delimiter=',' )
X_test = []
for row in datareader:
 X_test.append( row )

print( "reading y_test" )
y_test = []
dataReader = csv.reader( open( 'ddd22.csv' ), delimiter=',', quotechar='"' ) # csv file has to be in the same directory as this Python .py file
for row in dataReader:
 y_test.append( row )

X_train = np.asarray( X_train )
y_train = np.asarray( y_train )

print( "np.shape( X_train ) = ", np.shape( X_train ) )
print( "np.shape( y_train ) = ", np.shape( y_train ) )

X_test = np.asarray( X_test )
y_test = np.asarray( y_test )

print( "np.shape( X_test ) = ", np.shape( X_test ) )
print( "np.shape( y_test ) = ", np.shape( y_test ) )

X_train = X_train.reshape(3500,1000).astype('float32')
y_train = y_train.reshape(1,3500).astype('float32')

X_test = X_test.reshape(700,1000).astype('float32')
y_test = y_test.reshape(1,700).astype('float32')

print( "X_train.shape = ", X_train.shape )
print( "y_train.shape = ", y_train.shape )

print( "y_train[0:99] = ", y_train[0:99] )
print( "X_train[0] = ", X_train[0] )

print( "y_test[0] = ", y_test[0] )
print( "y_train[0] = ", y_train[0] )

max_X_train = np.amax( X_train )
max_y_train = np.amax( y_train )

X_train /= max_X_train
y_train /= max_y_train

max_X_test = np.amax( X_test )
max_y_test = np.amax( y_test )

X_test /= max_X_test
y_test /= max_y_test

print( "X_train[0] = ", X_train[0] )
print( "y_train[0:99] = ", y_train[0:99] )

y_train = y_train.flat
y_test = y_test.flat

print( "np.shape( X_train ) = ", np.shape( X_train ) )
print( "np.shape( y_train ) = ", np.shape( y_train ) )
print( "np.shape( X_test ) = ", np.shape( X_test ) )
print( "np.shape( y_test ) = ", np.shape( y_test ) )

# convert the given 0..29 or thus totally 1 + 29 = 30 classes into one-hot-encoded representations
# E.g. the single digit 5 becomes ( 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 )
# You do this for both the trainings output as well as for the test output.

n_classes = 29
y_train = keras.utils.to_categorical( y_train, n_classes)
y_test = keras.utils.to_categorical( y_test, n_classes)

print( "y_train[0] = ", y_train[0] )
print( "y_test[0] = ", y_test[0] )

model = Sequential()
model.add(Dense((64), activation='sigmoid', input_shape=(1000,)))
model.add(Dense((29), activation='softmax'))
model.summary()
model.compile( loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.fit( X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))

