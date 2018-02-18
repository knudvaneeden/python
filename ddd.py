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
#
from keras.callbacks import TensorBoard
#
# print( "reading X_train" )
#
datafile = open( 'ddd11.csv', 'r' )
datareader = csv.reader( datafile, delimiter=',' )
X_train = []
for row in datareader:
 X_train.append( row )
X_train = np.array( X_train )
X_train = X_train.reshape(3500,1000).astype('float32')
max_X_train = np.amax( X_train )
X_train /= max_X_train
#
# print( "reading y_train" )
#
y_train = []
dataReader = csv.reader( open( 'ddd12.csv' ), delimiter=',', quotechar='"' ) # csv file has to be in the same directory as this Python .py file
for row in dataReader:
 y_train.append( row )
y_train = np.array( y_train )
y_train = y_train.reshape(1,3500).astype('int')
y_train = y_train[ 0 ]
#
# print( "reading X_test" )
#
datafile = open( 'ddd21.csv', 'r' )
datareader = csv.reader( datafile, delimiter=',' )
X_test = []
for row in datareader:
 X_test.append( row )
X_test = np.array( X_test )
X_test = X_test.reshape(700,1000).astype('float32')
max_X_test = np.amax( X_test )
X_test /= max_X_test
#
# print( "reading y_test" )
#
y_test = []
dataReader = csv.reader( open( 'ddd22.csv' ), delimiter=',', quotechar='"' ) # csv file has to be in the same directory as this Python .py file
for row in dataReader:
 y_test.append( row )
y_test = np.array( y_test )
y_test = y_test.reshape(1,700).astype('int')
y_test = y_test[ 0 ]

# debug begin
print ( y_test ) # array
print( type( y_test ) ) # type
print( y_test.shape ) # dimensions
print( y_test.ndim ) # total number of dimensions
print( type( y_test ).__name__ ) # typicall returns ndarray
exit()
# debug end
#---------------

#
n_classes = 29
y_train = keras.utils.to_categorical( y_train, n_classes)
y_test = keras.utils.to_categorical( y_test, n_classes)
#
model = Sequential()
#
model.add(Dense((64), activation='relu', input_shape=(1000,)))
#
# model.add(Dense((64), activation='relu' )) #added extra hidden layer
# model.add(Dense((64), activation='relu' )) #added extra hidden layer
# model.add(Dense((64), activation='relu' )) #added extra hidden layer
#
# model.add(Dense((64), activation='tanh', input_shape=(1000,)))
# model.add(Dense((64), activation='sigmoid', input_shape=(1000,)))
#
model.add(Dense((29), activation='softmax'))
#
model.summary()
#
# model.compile( loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.compile( loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
#
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#
# model.fit( X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))
model.fit( X_train, y_train, batch_size=1000, epochs=20 )
#
classes = model.predict( X_test, batch_size = 1000 )
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
