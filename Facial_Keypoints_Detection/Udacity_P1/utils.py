"""
Convinence functions for facial keypoints dectection.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import keras

def load_data(file):
    """
    Load data from csv file to get image names and keypoints array.
    Then use image names to get image arrays from images.
    Args:
        file -- String, csv file path
    Return:
        images -- List of numpy arrays, each one has different width and height
        keypoints -- Numpy array of shape (N, 68, 2)
    """
    # Read csv file
    df = pd.read_csv(file)
    
    # Get image arrays
    image_names = df.iloc[:, 0]
    image_names = image_names.as_matrix()
    images = []
    for image in image_names:
        image_path = 'data/training/' + image
        image_array = mpimg.imread(image_path)
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        images.append(image_array)
        
    # Get key points
    keypoints = df.iloc[:, 1:].as_matrix().reshape(-1, 68, 2)
    
    return images, keypoints

def rescale(image, key_points, smaller_len=250, grayscale=True):
    """
    First rescale images' smaller length to smaller_len while reserving image ratio constant.
    Then adjust keypoints' coordinates.
    Args:
        image -- Numpy array of shape (H, W, C)
        keypoints -- Numpy array of shape (68, 2)
        smaller_len -- Integer, smaller length of rescaled image
        grayscale -- Boolean, whether to convert to grayscale image
    Return:
        image_resized -- Numpy arrays, rescaled image of same shape as input image
        keypoints -- Numpy array of same shape as input keypoints
    """
    # Convert to gray image
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize samller length to 250 while reserving image ratio
    height, width = image.shape[:2]
    ratio = smaller_len / np.minimum(height, width)
    new_height = np.int(np.round(height * ratio))
    new_width = np.int(np.round(width * ratio))
    image_resized = cv2.resize(image, (new_width, new_height))
    
    # Rescale kep points
    keypoints = key_points * np.asarray([new_width / width, new_height / height], np.float64)
    
    return image_resized, keypoints

def crop(image, key_points, new_height=224, new_width=224, grayscale=True):
    """
    First rescale images' smaller length to smaller_len while reserving image ratio constant.
    Then adjust keypoints' coordinates.
    Args:
        image -- Numpy array of shape (H, W, C) or (H, W)
        keypoints -- Numpy array of shape (68, 2)
        new_height -- Integer, height of the new image
        new_width -- Integer, width of the new image
        grayscale -- Boolean, indicating input image is grayscale or not
    Return:
        image_cropped -- Numpy arrays, rescaled image of same shape as input image
        keypoints -- Numpy array of same shape as input keypoints
    """
    # Crop image to (new_height, new_width)
    height, width = image.shape[:2]
    top_start = np.random.randint(height - new_height)
    left_start = np.random.randint(width - new_width)
    if grayscale:
        image_cropped = image[top_start:top_start+224, left_start:left_start+224]
    else:
        image_cropped = image[top_start:top_start+224, left_start:left_start+224, :]
    
    # Modify coordinates of key points
    keypoints -= np.asarray([left_start, top_start])
    
    return image_cropped, keypoints

def preprocess(file):
    """
    First rescale images while reserving ratio constant.
    Then crop images to a certain size.
    Finally delete images which have keypoints out of boarder.
    Args:
        file -- csv file path
    Return:
        images -- Numpy arrays of shape (N, H, W, C) or (N, H, W), values in range [0, 1]
        keypoints -- Numpy array of shape (N, 68, 2)
    """
    images, key_points = load_data(file)
    
    images_processed, key_points_processed = [], []
    for i in range(len(images)):
        image, kpts = images[i], key_points[i]
        image, kpts = rescale(image, kpts, grayscale=False)
        image, kpts = crop(image, kpts, grayscale=False)
        images_processed.append(image[np.newaxis, ...])
        key_points_processed.append(kpts[np.newaxis, ...])
        
    images = np.concatenate(images_processed) / 255.0
    key_points = np.concatenate(key_points_processed)
    
    del_i = []
    for i in range(len(key_points)):
        if (key_points[i] > 224).any() or (key_points[i] < 0).any():
            del_i.append(i)
    images = np.delete(images, del_i, axis=0)
    key_points = np.delete(key_points, del_i, axis=0)
    key_points = (key_points - 100) / 50
    
    return images, key_points

def visualize_preds(test_images, fixed_test_labels, fixed_preds):
    """
    Show a random image and its key points
    Args:
        test_images -- Numpy arrays of test images
        fixed_test_labels -- Unnormalized numpy arrays of test key points
        fixed_preds -- Unnormalized numpy arrays of predictions
    """
    idx = np.random.randint(fixed_preds.shape[0])
    plt.imshow(test_images[idx])
    plt.scatter(fixed_preds[idx][:, 0], fixed_preds[idx][:, 1], c='g', s=20, marker='.')
    plt.scatter(fixed_test_labels[idx][:, 0],
                fixed_test_labels[idx][:, 1],
                c='r', s=20, marker='.')
    plt.show()
    
def extract_features():
    # Load vgg16 model
    vgg16_model = keras.applications.VGG16(include_top=False, weights='imagenet')
    
    # Load training images and extract features
    # Becareful of batch_size! May cause memory error...
    training_images = np.load('data/training_images_del_rgb.npy')
    extracted_training_features = vgg16_model.predict(training_images, batch_size=64)
    np.save('data/extracted_training_features_rgb.npy', extracted_training_features)
    
    # Load test images and extract features
    test_images = np.load('data/test_images_del_rgb.npy')
    extracted_test_features = vgg16_model.predict(test_images, batch_size=64)
    np.save('data/extracted_test_features_rgb.npy', extracted_test_features)

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