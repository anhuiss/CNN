import keras
import numpy as np
import matplotlib.pyplot as plt

def get_cifar10(batch_size=128, center=True, normalization=True, data_augmentation=True):
    """
    Get cifar10 data from keras.datasets
    Args:
        batch_size -- Integer, number of samples in a single batch
        center -- Boolean, whether to subtract cifar10 data by per-channel mean
        normalization -- Boolean, whether to divide cifar10 data by 255
        data_augmentation -- Boolean, whether to use keras's convinence data augmentation tool
    Return:
        train_generator, X_test, y_test for using data augmentation
        (X_train, y_train), (X_test, y_test) for not using data augmentation
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    if normalization:
        X_train /= 255
        X_test /= 255
    
    if center:
        mean = np.mean(X_train, axis=(0, 1, 2), dtype=np.float32, keepdims=True)
        X_train -= mean
        X_test -= mean
        
    if data_augmentation:
        train_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=0.2,
                                                                     width_shift_range=0.2,
                                                                     height_shift_range=0.2,
                                                                     shear_range=0.2,
                                                                     zoom_range=0.2,
                                                                     horizontal_flip=True)
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        
        return train_generator, X_test, y_test
    else:
        return (X_train, y_train), (X_test, y_test)

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 3e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


class History(keras.callbacks.Callback):
    """
    Create a custom Callback instance to record loss and accuracy for
    training and validation, also including a plot function to plot
    loss and accuracy curves.
    """
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch':[]}
        self.accuracy = {'batch': [], 'epoch':[]}
        self.val_losses = {'batch': [], 'epoch':[]}
        self.val_accuracy = {'batch': [], 'epoch':[]}
        
    def on_batch_end(self, batch, logs={}):
        # logs -- dict_keys(['batch', 'size', 'loss', 'acc'])
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        
    def on_epoch_end(self, epoch, logs={}):
        # logs -- dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_losses['epoch'].append(logs.get('val_loss'))
        self.val_accuracy['epoch'].append(logs.get('val_acc'))
        
    def plot(self):
        plt.figure(figsize=(15, 10))

        # Plot loss curve over training batches
        plt.subplot(221)
        plt.plot(self.losses['batch'], label='train_loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        # Plot accuracy curve over training batches
        plt.subplot(222)
        plt.plot(self.accuracy['batch'], label='train_acc')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        # Plot loss curves over training and validation epochs
        plt.subplot(223)
        plt.plot(self.losses['epoch'], label='train_loss')
        plt.plot(self.val_losses['epoch'], label='valid_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        # Plot accuracy curves over training and validation epochs
        plt.subplot(224)
        plt.plot(self.accuracy['epoch'], label='train_acc')
        plt.plot(self.val_accuracy['epoch'], label='valid_acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

