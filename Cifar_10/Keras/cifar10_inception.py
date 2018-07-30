"""
Try training cifar10 using a simple inception network simialr to
the following model architecture:
    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842
    arXiv:1409.4842 [cs.CV]
"""
import keras
from Inception import inception_v1
from cifar10_utils import History, get_cifar10

# Training parameters
LEARNING_RATE = 1e-3
DATA_AUGMENTATION = True  # reduce overfitting
BATCH_SIZE = 128
EPOCHS = 200
KERNEL_INITIALIZER = 'he_normal'

def main():
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
    
    model = inception_v1(input_shape=(32, 32, 3), n_classes=10, initializer=KERNEL_INITIALIZER)
    model.summary()
    optimizer = keras.optimizers.Adam(LEARNING_RATE, decay=1e-6)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Prepare callbacks for model saving and for learning rate adjustment.
    hists = History()
    es = keras.callbacks.EarlyStopping(monitor='val_acc', patience=15)
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                   factor=0.1,
                                                   cooldown=0,
                                                   patience=10,
                                                   verbose=1)
    callbacks = [hists]
    
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
    main()
