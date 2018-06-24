'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def __loadData():
    img_rows, img_cols = 28, 28
    num_classes = 10
    (X, Y), (testX, testY) = mnist.load_data()
    Y = keras.utils.to_categorical(Y, num_classes)
    testY = keras.utils.to_categorical(testY, num_classes)
    if K.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
    X = X.astype('float32')
    testX = testX.astype('float32')
    return X, Y, testX, testY


# this function will take the dimenstion of the input image as the parameters.
def __defineNetwork(input_shape) -> object:
    num_classes = 10
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


def __fitParameters():
    X, Y, testX, testY = __loadData()
    print(X.shape[1:])
    network = __defineNetwork(X.shape[1:])
    epochs = 12
    batch_size = 64
    network.fit(X, Y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(testX, testY))
    score = network.evaluate(testX, testY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# exposed function....
def apply():
    __fitParameters()
