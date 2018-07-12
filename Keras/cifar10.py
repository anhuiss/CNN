"""
Training cifar10 using keras.layers functional APIs
"""
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense, Activation, Dropout
from cifar10_utils import get_cifar10, History, lr_schedule

# Training parameters
LEARNING_RATE = 3e-3
DATA_AUGMENTATION = True  # reduce overfitting
BATCH_SIZE = 128
EPOCHS = 200
BATCH_NORMALIZATION = True
REGULARIZATION = 0.0
KERNEL_INITIALIZER = 'he_normal'

def my_model(reg, bn=True):
    """
    Create a keras.Model instance
    Model architecture:
        conv--(bn)--relu--conv--(bn)--relu--maxpool
        conv--(bn)--relu--conv--(bn)--relu--maxpool
        fc--(bn)--relu--dropout--fc
    """
    # Input
    X_input = keras.layers.Input(shape=(32, 32, 3), name='input')
    
    # Block 1
    X = Conv2D(filters=32, kernel_size=3, padding='same',
               kernel_initializer=KERNEL_INITIALIZER,
               kernel_regularizer=tf.keras.regularizers.l2(reg),
               name='block1_conv1')(X_input)
    if bn:
        X = BatchNormalization(axis=-1, name='block1_bn1')(X)
    X = Activation('relu', name='block1_relu1')(X)
    X = Conv2D(filters=32, kernel_size=3, padding='same',
               kernel_initializer=KERNEL_INITIALIZER,
               kernel_regularizer=tf.keras.regularizers.l2(reg),
               name='block1_conv2')(X)
    if bn:
        X = BatchNormalization(axis=-1, name='block1_bn2')(X)
    X = Activation('relu', name='block1_relu2')(X)
    X = MaxPooling2D(pool_size=2, strides=2, name='block1_maxpool')(X)
    
    # Block 2
    X = Conv2D(filters=64, kernel_size=3, padding='same',
               kernel_initializer=KERNEL_INITIALIZER,
               kernel_regularizer=tf.keras.regularizers.l2(reg),
               name='block2_conv1')(X)
    if bn:
        X = BatchNormalization(axis=-1, name='block2_bn1')(X)
    X = Activation('relu', name='block2_relu1')(X)
    X = Conv2D(filters=64, kernel_size=3, padding='same',
               kernel_initializer=KERNEL_INITIALIZER,
               kernel_regularizer=tf.keras.regularizers.l2(reg),
               name='block2_conv2')(X)
    if bn:
        X = BatchNormalization(axis=-1, name='block2_bn2')(X)
    X = Activation('relu', name='block2_relu2')(X)
    X = MaxPooling2D(pool_size=2, strides=2, name='block2_maxpool')(X)
    
    # Flatten
    X = Flatten()(X)
    # Dense layer
    X = Dense(units=512,
              kernel_initializer=KERNEL_INITIALIZER,
              kernel_regularizer=tf.keras.regularizers.l2(reg))(X)
    if bn:
        X = BatchNormalization(axis=-1, name='dense_bn1')(X)
    X = Activation('relu', name='dense_relu1')(X)
    X = Dropout(0.5)(X)
    X = Dense(units=10, activation='softmax',
              kernel_initializer=KERNEL_INITIALIZER,
              kernel_regularizer=tf.keras.regularizers.l2(reg))(X)
    
    model = keras.Model(inputs=X_input, outputs=X)
    
    return model

def main(argv):
    if DATA_AUGMENTATION:
        print('Using data augmentation.')
        train_generator, X_test, y_test = get_cifar10(
                batch_size=BATCH_SIZE,
                center=True,
                normalization=True,
                data_augmentation=True)
    else:
        print('Not using data augmentation.')
        (X_train, y_train), (X_test, y_test) = get_cifar10(
                batch_size=BATCH_SIZE,
                center=True,
                normalization=True,
                data_augmentation=False)
    
    # Create a keras model
    model = my_model(reg=REGULARIZATION, bn=BATCH_NORMALIZATION)
    opt = keras.optimizers.Adam(LEARNING_RATE, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    hists = History()
    es = keras.callbacks.EarlyStopping(monitor='val_acc', patience=15)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, verbose=1)
    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    callbacks = [hists, lr_scheduler]
    
    if DATA_AUGMENTATION:
        model.fit_generator(train_generator,
                            epochs=EPOCHS,
                            validation_data=(X_test, y_test),
                            callbacks=callbacks)
    else:
        model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(X_test, y_test),
                  callbacks=callbacks)
    
    hists.plot()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)