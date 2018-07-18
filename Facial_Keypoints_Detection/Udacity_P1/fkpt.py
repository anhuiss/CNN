import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from utils import *

def fast_layers(X, filters, kernel_size, strides, kernel_initializer, activation, stage, dropout_rate):
    """
    Convinence function for conv-maxpool-dropout layers.
    Args:
        X -- Input
        filters -- Integer, number of filters of conv2d layer
        kernel_size -- Integer or tuple of 2 integers, kernel size of conv2d layer
        strides -- Integer or tuple of 2 integers, kernel strides of conv2d layer
        kernel_initializer -- String or keras.initializers instance
        activation -- String or keras.activations instance
        stage -- Integer
        dropout_rate -- Float in range [0, 1], indicating dropout rate in Dropout layer
    Return:
        X -- Same shape and dtype as input
    """
    X = Conv2D(filters,
               kernel_size=kernel_size,
               strides=strides,
               kernel_initializer=kernel_initializer,
               activation=activation,
               name='conv'+str(stage))(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = Dropout(dropout_rate)(X)
    
    return X

def my_model():
    # Input layer
    X_input = keras.Input(shape=(224, 224, 1))
    # Conv layers
    X = fast_layers(X_input, 32, (4, 4), (1, 1), keras.initializers.random_normal(), 'relu', 1, 0.1)
    X = fast_layers(X, 64, (3, 3), (1, 1), keras.initializers.random_normal(), 'relu', 2, 0.2)
    X = fast_layers(X, 128, (2, 2), (1, 1), keras.initializers.random_normal(), 'relu', 3, 0.3)
    X = fast_layers(X, 256, (4, 4), (1, 1), keras.initializers.random_normal(), 'relu', 4, 0.4)
    # Flatten
    X = Flatten()(X)
    # Fully-connected layers
    X = Dense(512, kernel_initializer=keras.initializers.glorot_uniform(), activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)
#     X = Dense(1000, kernel_initializer=keras.initializers.glorot_uniform(), activation='relu', name='fc2')(X)
#     X = Dropout(0.6)(X)
    X = Dense(136, kernel_initializer=keras.initializers.glorot_uniform(), activation=None, name='fc3')(X)
    X = Reshape((68, 2))(X)
    
    # Construct a model
    model = keras.Model(inputs=X_input, outputs=X)
    
    return model

def main():
    # Load data
    training_images = np.load('data/training_images_del.npy')
    training_kpts = np.load('data/training_kpts_del.npy')
    test_images = np.load('data/test_images_del.npy')
    test_kpts = np.load('data/test_kpts_del.npy')
    training_images = training_images[..., np.newaxis]
    test_images = test_images[..., np.newaxis]
    
    # Training
    model = my_model()
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    hists = History()
    model.fit(training_images, training_kpts, batch_size=128, epochs=100, callbacks=[hists])
    print('Test lost: %.4f' % model.evaluate(test_images, test_kpts))
    hists.plot()
    
    # Visualize predictions
    preds = model.predict(test_images, batch_size=128)
    fixed_preds = preds * 50 + 100
    fixed_test_kpts = test_kpts * 50 + 100
    visualize_preds(test_images, fixed_test_kpts, fixed_preds)

