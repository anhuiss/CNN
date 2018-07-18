import keras
import numpy as np
import matplotlib.pyplot as plt
from utils import root_mean_squared_error, History
from utils import lenet_model, my_model

def main():
    # Load augmented images and keypoints
    training_images = np.load('data/training_images_augs.npy')
    training_labels = np.load('data/training_keypoints_augs.npy')
    valid_images = np.load('data/valid_images_augs.npy')
    valid_labels = np.load('data/training_keypoints_augs.npy')
    
    # divide by 255
    training_images = training_images / 255
    valid_images = training_images / 255
    
    # Construct a model
    model = my_model()
    
    # Compile
#    opt = keras.optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True)
    opt = keras.optimizers.Adam(lr=1e-4, decay=1e-6)
    model.compile(optimizer=opt, loss=root_mean_squared_error)
    
    # Callbacks
    save_dir = 'tmp/saved_model/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    hists = History()
    ckpt = keras.callbacks.ModelCheckpoint(save_dir,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           period=25)
    
    callbacks=[hists, ckpt]
    
    # Training
    model.fit(training_images,
              training_labels,
              batch_size=64,
              epochs=5,
              validation_data=(valid_images, valid_labels),
              callbacks=callbacks)
    # Plot loss curves
    hists.plot()
    
    # Visualize predictions
    test_images = np.load('data/test_images.npy') / 255
    preds = model.predict(test_images[..., np.newaxis], batch_size=256)
    idx = np.random.randint(preds.shape[0])
    plt.imshow(test_images[idx], cmap='gray')
    plt.scatter(preds[idx][:, 0], preds[idx][:, 1], c='r', s=20, marker='.')
    plt.show()
    
if __name__ == '__main__':
    main()
    
