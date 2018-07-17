"""
Training cifar10 using resnet_20 simialr to the following model architecture:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
"""
import keras
from keras.regularizers import l2
from ResNet_20 import ResNet_20
from cifar10_utils import get_cifar10, History

# Training parameters
LEARNING_RATE = 1e-3
DATA_AUGMENTATION = True  # reduce overfitting
REGULARIZATION = 0.0
BATCH_SIZE = 128
EPOCHS = 10

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
    
    # Create a ResNet-20 model
    model = ResNet_20(input_shape=(32, 32, 3),
                      n_classes=10,
                      data_format='channels_last',
                      initializer='he_normal',
                      regularizer=l2(REGULARIZATION))
    
    opt = keras.optimizers.Adam(LEARNING_RATE, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    hists = History()
    es = keras.callbacks.EarlyStopping(monitor='val_acc', patience=15)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, verbose=1)
    callbacks = [hists, es, reduce_lr]
    
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