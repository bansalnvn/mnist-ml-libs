import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist


def __loadData():
    X, Y, testX, testY = mnist.load_data(one_hot=True)
    X = X.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])
    return X, Y, testX, testY


# this function will take the dimenstion of the input image as the parameters.
def __defineNetwork(width: int, height: int, depth: int) -> object:
    network = input_data(shape=[None, width, height, depth], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')
    return network


def __fitParameters():
    X, Y, testX, testY = __loadData()
    network = __defineNetwork(28, 28, 1)
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': X}, {'target': Y}, n_epoch=20,
              validation_set=({'input': testX}, {'target': testY}),
              snapshot_step=100, show_metric=True, run_id='convnet_mnist')
    model.score = model.evaluate(testX, testY)
    print('Test accuracy:', model.score[0] * 100)


# exposed function....
def apply():
    __fitParameters()
