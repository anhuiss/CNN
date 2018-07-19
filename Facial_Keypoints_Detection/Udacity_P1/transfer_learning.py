"""
Try to use features extracted from vgg16 to detect facial keypoints.
"""
import keras
import numpy as np
from keras.layers import Flatten, Dense, Dropout, Reshape
from utils import *

def my_model():
    # Input
    X_inputs = keras.Input(shape=(7, 7, 512))
    
    # Dense layers
    X = Flatten()(X_inputs)
    X = Dense(1024, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=keras.regularizers.l2(1e-3))(X)
    X = Dropout(0.5)(X)
    X = Dense(1024, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=keras.regularizers.l2(1e-3))(X)
    X = Dropout(0.5)(X)
    X = Dense(136)(X)
    X = Reshape((68, 2))(X)
    
    # Construct a model
    model = keras.Model(inputs=X_inputs, outputs=X)
    
    return model

def main():
    # Load data
    training_features = np.load('data/extracted_training_features_rgb.npy')
    training_labels = np.load('data/training_kpts_del_rgb.npy')
    test_features = np.load('data/extracted_test_features_rgb.npy')
    test_labels = np.load('data/test_kpts_del_rgb.npy')
    
    model = my_model()
    
#    opt = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)
    opt = keras.optimizers.Adam(lr=1e-3, decay=1e-6)
    model.compile(optimizer=opt, loss='mean_squared_error')
    
    # Create save directory if not exists
    model_save_dir = 'tmp/transfer_learning/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    
    # Callbacks
    hists = History()
    ckpt = keras.callbacks.ModelCheckpoint(model_save_dir,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           period=25)
    callbacks=[hists, ckpt]
    
    # Training
    model.fit(training_features,
              training_labels,
              batch_size=128,
              epochs=5,
              validation_data=(test_features, test_labels),
              callbacks=callbacks)
    hists.plot()
    
    # Visualize predictions
    preds = model.predict(test_features, batch_size=128)
    fixed_preds = preds * 50 + 100
    fixed_test_labels = test_labels * 50 + 100
    test_images = np.load('data/test_images_del_rgb.npy')
    visualize_preds(test_images, fixed_test_labels, fixed_preds)

if __name__ == '__main__':
    main()


    
    
    
    
    
    