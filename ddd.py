import csv
import keras
import numpy as np
np.random.seed(42)
from keras.datasets import mnist
from keras.datasets import mnist
from time import time
#
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from keras.callbacks import TensorBoard
#
#
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
# y_train /= max_y_train

max_X_test = np.amax( X_test )
max_y_test = np.amax( y_test )

X_test /= max_X_test
# y_test /= max_y_test

print( "X_train[0] = ", X_train[0] )
print( "y_train[0:99] = ", y_train[0:99] )













y_train = y_train.flat
y_test = y_test.flat

print( "np.shape( X_train ) = ", np.shape( X_train ) )
print( "np.shape( y_train ) = ", np.shape( y_train ) )
print( "np.shape( X_test ) = ", np.shape( X_test ) )
print( "np.shape( y_test ) = ", np.shape( y_test ) )

# convert the given 0..28 or thus totally 1 + 28 = 29 classes into one-hot-encoded representations
# E.g. the single digit 5 becomes ( 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 )
# You do this for both the trainings output as well as for the test output.

n_classes = 29
y_train = keras.utils.to_categorical( y_train, n_classes)
y_test = keras.utils.to_categorical( y_test, n_classes)

print( "y_test[0] = ", y_test[0] )
print( "y_train[0] = ", y_train[0] )

model = Sequential()
#
# 1 hidden layer (with 64 sigmoid neurons)
#
# possible neuron types: perceptron (can not be chosen), options to choose from: sigmoid (output from 0 to 1), tanh (output from -1 to 1), relu (output from 0 to max( 0, x ))
#
model.add(Dense((64), activation='relu', input_shape=(1000,)))
# model.add(Dense((64), activation='tanh', input_shape=(1000,)))
# model.add(Dense((64), activation='sigmoid', input_shape=(1000,)))
#
# The softmax function represents a probability
# E.g. for the xth output calculated as e^x / sum( k = 1 to n of e^k )
# The result is that small values tend to go to 0, large values tend to go to 1.
#
# 1 output layer (with 10 neurons, representing the digits 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
#
model.add(Dense((29), activation='softmax'))
#
# you see here for the first (=input layer) 784 x 64 + 64 = 50240 parameters = total amount of neurons in the input layer times ( total amount + 1 ) of neurons in the hidden layer parameters
#
# you see here for the second layer (=hidden layer) 64 x 10 + 10 = 650 parameters = total amount of neurons in the hidden layer times ( total amount + 1 ) of neurons in the output layer parameters
#
# dense_1 (Dense)              (None, 64)                50240
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                650
#
model.summary()
#
# possibility here to choose from as the cost function are:
# 'mean_squared_error' (also called the mean quadratic cost function)
# and
# 'categorical_crossentropy' (designed to let it grow fast with large cost function values, and much less when small difference in the cost function values)
#
# here 'lr' means 'l'earning 'r'ate, you can also very this from small to larger to see the effect on the calculations.
#
# here 'SGD' means 'S'tochastic 'G'radient 'D'escent
#
# model.compile( loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.compile( loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
#
# log file for TensorBoard
#
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#
# in general choosing 'relu' and 'categorical_crossentropy' gives the fastest learning. You can also vary the 'learning rate' (=lr))
#
model.fit( X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))
#
# generate predictions on new data // see https://keras.io
#
classes = model.predict( X_test, batch_size = 128 )
print( classes )
#
# The highest value is to be found in the entry of number 1. So that is the prediction.
#
# Change e.g.:
#
# * more iterations (=epochs),
# * other function instead of sigmoid (=relu, tanh, ...)
# * another cost function (e.g. cross entropy)
#
# e.g. disable / enable the relevant lines in the source code above (relu, ...).
#
def FNArrayGetIndexMaximumI( list ):
 maximum = 0
 value = 0
 I = 0
 indexI = 0
 for I, value in enumerate( list ):
  if ( value > maximum ):
   maximum = value
   indexI = I
 return( indexI )
#
# check the first predicted values
#
print( " " )
print( classes[0] )
print( y_test[0] )
print( np.amax( classes[0] ) )
print( "the prediction of the neural network is that the most probable result = ", FNArrayGetIndexMaximumI( classes[ 0 ] ) )
#
print( " " )
print( classes[1] )
print( y_test[1] )
print( np.amax( classes[1] ) )
print( "the prediction of the neural network is that the most probable result = ", FNArrayGetIndexMaximumI( classes[ 1 ] ) )
#
print( " " )
print( classes[2] )
print( y_test[2] )
print( np.amax( classes[2] ) )
print( "the prediction of the neural network is that the most probable result = ", FNArrayGetIndexMaximumI( classes[ 2 ] ) )
#
# should be 12
#
print( " " )
print( classes[31] )
print( y_test[31] )
print( np.amax( classes[31] ) )
print( "the prediction of the neural network is that the most probable result = ", FNArrayGetIndexMaximumI( classes[ 31 ] ) )
#
for I in range( 0, 40 ):
 print ( I )
 print( " " )
 print( classes[ I ] )
 print( y_test[ I ] )
 print( np.amax( classes[ I ] ) )
 print( "the prediction of the neural network is that the most probable result = ", FNArrayGetIndexMaximumI( classes[ I ] ) )
