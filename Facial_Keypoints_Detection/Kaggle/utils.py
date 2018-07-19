"""
Convinence functions for facial keypoints detection.
"""
import pandas as pd
import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Dropout
import keras.backend as K
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa

def process_data(file, is_training=True):
    """
    Convinence function for conv-maxpool-dropout layers.
    Args:
        file -- String, training file or test file path
        is_training -- Boolean, indicating which file to process, training file or test file
    Return:
        images -- Numpy array, shape is (N, 96, 96)
        keypoints -- Numpy array, returned when is_training is True, shape is (N, 15, 2)
    """
    # Load csv file
    df = pd.read_csv(file)
    # Delete n.a. data
    df = df.dropna()
    
    # Get image arrays
    images = df['Image'].as_matrix()
    images = [np.fromstring(image, dtype=np.uint8, count=96*96, sep=' ') for image in images]
    images = np.concatenate(images).reshape(-1, 96, 96)
    
    # Get training image key points
    if is_training:
        keypoints = df.drop('Image', axis=1).as_matrix().reshape(-1, 15, 2)
        return images, keypoints 
    
    return images

def flip_images_and_keypoints(images, keypoints):
    """
    Flip images and keypoints horizontally.
    Args:
        images -- Numpy array of shape (N, H, W, C)
        keypoints -- Numpy array of shape (N, 15, 2)
    Return:
        images_flipped -- Numpy array, same shape as images
        keypoints_flipped -- Numpy array, same shape as keypoints
    """
    images_flipped = np.flip(images, axis=2)
    keypoints_flipped = (np.asarray([96, 0]) - keypoints) * np.asarray([1, -1])
    
    return images_flipped, keypoints_flipped

def rotate_images_and_keypoints(images, keypoints, rotate_angle):
    """
    Rotate grayscale images and keypoints using 'imgaug' library.
    Reference:
        https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
    Args:
        images -- Numpy array of shape (N, H, W, 1)
        keypoints -- Numpy array of shape (N, 15, 2)
        rotate_angle -- Integer, rotate degree
    Return:
        images_rotated -- Numpy array, same shape as images
        keypoints_rotated -- Numpy array, same shape as keypoints
    """
    images = images[..., 0]

    keypoints_list = []
    for kpts in keypoints:
        keylst = []
        for k in kpts:
            keylst.append(ia.Keypoint(k[0], k[1]))
        keylst = ia.KeypointsOnImage(keylst, (96, 96))
        keypoints_list.append(keylst)
    
    mean_training_pixel = np.mean(images) # images are padded with mean pixel value after rotating
    seq = iaa.Sequential([iaa.Affine(rotate=rotate_angle, cval=mean_training_pixel)])
    seq_det = seq.to_deterministic()
    
    images_rotated = seq_det.augment_images(images)
    keypoints_rot = seq_det.augment_keypoints(keypoints_list)
    keypoints_rot_list = [k.get_coords_array()[np.newaxis, ...] for k in keypoints_rot]
    keypoints_rotated = np.concatenate(keypoints_rot_list)
    
    return images_rotated, keypoints_rotated
    
def aug_images_and_keypoints(images, keypoints):
    """
    Flip and rotate grayscale images and keypoints, then concatenate them into training
    and validation data.
    Args:
        images -- Numpy array of shape (N, H, W, 1)
        keypoints -- Numpy array of shape (N, 15, 2)
    Return:
        training_images_aug -- Numpy array of shape (6*N-1800, H, W, 1)
        training_keypoints_aug -- Numpy array of shape (6*N-1800, 15, 2)
        valid_images_aug -- Numpy array of shape (1800, 15, 2)
        valid_keypoints_aug -- Numpy array of shape (1800, 15, 2)
    """
    images_fl, keypoints_fl = flip_images_and_keypoints(images, keypoints)
    images_rot, keypoints_rot = rotate_images_and_keypoints(images, keypoints, 5)
    images_fl_rot, keypoints_fl_rot = rotate_images_and_keypoints(images_fl, keypoints_fl, 5)
    images_rot_2, keypoints_rot_2 = rotate_images_and_keypoints(images, keypoints, -5)
    images_fl_rot_2, keypoints_fl_rot_2 = rotate_images_and_keypoints(images_fl, keypoints_fl, -5)
    
    n_images_per_set = 2140
    n_valid = 300
    
    training_images_aug = []
    training_keypoints_aug = []
    valid_images_aug = []
    valid_keypoints_aug = []
    
    image_sets = (images, images_fl, images_rot[..., np.newaxis],
                  images_fl_rot[..., np.newaxis], images_rot_2[..., np.newaxis],
                  images_fl_rot_2[..., np.newaxis])
    keypoints_sets = (keypoints, keypoints_fl, keypoints_rot, keypoints_fl_rot,
                      keypoints_rot_2, keypoints_fl_rot_2)
    
    for ds in list(zip(image_sets, keypoints_sets)):
        mask = np.arange(n_images_per_set)
        np.random.shuffle(mask)
        
        training_images_aug.append(ds[0][mask[:-n_valid]])
        training_keypoints_aug.append(ds[1][mask[:-n_valid]])
        valid_images_aug.append(ds[0][mask[-n_valid:]])
        valid_keypoints_aug.append(ds[1][mask[-n_valid:]])
        
    training_images_aug = np.concatenate(training_images_aug)
    training_keypoints_aug = np.concatenate(training_keypoints_aug)
    valid_images_aug = np.concatenate(valid_images_aug)
    valid_keypoints_aug = np.concatenate(valid_keypoints_aug)
    
    return (training_images_aug, training_keypoints_aug), (valid_images_aug, valid_keypoints_aug)

class History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch':[]}
        self.val_losses = {'batch': [], 'epoch':[]}
        
    def on_batch_end(self, batch, logs={}):
        # logs -- dict_keys(['batch', 'size', 'loss', 'acc'])
        self.losses['batch'].append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        # logs -- dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
        self.losses['epoch'].append(logs.get('loss'))
        self.val_losses['epoch'].append(logs.get('val_loss'))
        
    def plot(self):
        plt.figure(figsize=(15, 10))

        plt.subplot(121)
        plt.plot(self.losses['batch'], label='train_loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        # Plot losses of epochs
        plt.subplot(122)
        plt.plot(self.losses['epoch'], label='train_loss')
        plt.plot(self.val_losses['epoch'], label='valid_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        
        plt.show()
    
def root_mean_squared_error(y_true, y_pred):
    """Custom loss function for model.compile"""
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def lenet_model():
    """
    Reference:
        http://cs231n.stanford.edu/reports/2016/pdfs/010_Report.pdf
    """
    model = keras.Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(96, 96, 1), name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name='pool1'))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name='pool2'))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name='pool3'))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name='pool4'))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv5'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name='pool5'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(500, activation='relu', name='dense1'))
    model.add(Dense(500, activation='relu', name='dense2'))
    model.add(Dense(500, activation='relu', name='dense3'))
    model.add(Dense(500, activation='relu', name='dense4'))
    model.add(Dense(30, name='dense5'))
    model.add(Reshape((15, 2), name='reshape'))
    
    return model

def my_model():
    """
    Reference:
        http://cs231n.stanford.edu/reports/2016/pdfs/010_Report.pdf
    """
    model = keras.Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(96, 96, 1),
                     kernel_initializer='he_normal', name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name='pool1'))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_normal', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name='pool2'))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_normal', name='conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name='pool3'))
    model.add(Conv2D(128, (3, 3), activation='relu',
                     kernel_initializer='he_normal', name='conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name='pool4'))
    model.add(Conv2D(256, (3, 3), activation='relu',
                     kernel_initializer='he_normal', name='conv5'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name='pool5'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal', name='dense2'))
    model.add(Dropout(0.5))
    model.add(Dense(30, kernel_initializer='he_normal', name='dense3'))
    model.add(Reshape((15, 2), name='reshape'))
    
    return model