# -*- coding: utf-8 -*-
import numpy as np
import time
import keras
from cifar10 import my_model
from Inception import inception_v1
from cifar10_utils import get_cifar10

def random_search(learning_rates, num_epochs, threshold=0.42, data_aug=True):
    """
    Random search learning rates
    Args:
        learning_rates -- List of floats
        num_epochs -- Boolean, number of epochs to run
        threshold -- Float, whether to continue training after the first epoch
    Return:
        Just print best
    """
    best_val_acc = -1
    best_lr = -1
    val_accs = []
    lrs = []
    
    if data_aug:
        train_generator, X_test, y_test = get_cifar10()
    else:
        X_train, y_train, X_test, y_test = get_cifar10(data_augmentation=False)
    
    for count, lr in enumerate(learning_rates):
        start = time.time()
        print('Count:', count+1)
        print('learning_rate: {:.4e}'.format(lr))
        
        # Create a model
        model = inception_v1((32, 32, 3), 10)
        opt = keras.optimizers.Adam(lr, decay=1e-6)
        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        if data_aug:
            h = model.fit_generator(train_generator, epochs=1, verbose=1)
            if h.history['acc'][0] > threshold:
                model.fit_generator(train_generator, epochs=num_epochs-1, verbose=1)
        else:
            h = model.fit(X_train, y_train, batch_size=128, epochs=1, verbose=1)
            if h.history['acc'][0] > threshold:
                h = model.fit(X_train, y_train, batch_size=128, epochs=num_epochs-1, verbose=1)
        
        val_res = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
        print('Validation loss: %.4f, accuracy: %.4f' % (val_res[0], val_res[1]))
        if best_val_acc < val_res[1]:
            best_val_acc = val_res[1]
            best_lr = lr
        val_accs.append(val_res[1])  
        lrs.append(lr)
        print('Time: %.2f\n' % (time.time()-start))
    print('Best val_accuracy: %.4f, learning_rate: %.4e\n' % (best_val_acc, best_lr))
    
    for item in sorted(list(zip(lrs, val_accs)), key=lambda x: x[1], reverse=True):
        print('learning_rate: %.4e, val_accuracy: %.4f' % (item[0], item[1]))

def main():
    learning_rates = [10**(np.random.random() * (-1) - 2) for _ in range(10)]
    random_search(learning_rates, num_epochs=5, threshold=0.33)
    
if __name__ == '__main__':
    main()
