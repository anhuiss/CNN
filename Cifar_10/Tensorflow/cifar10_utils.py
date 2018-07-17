import tensorflow as tf
import numpy as np

TRAIN_FILES = ['data/cifar-10-batches-bin/data_batch_' + str(i) + '.bin' for i in range(1, 6)]
TEST_FILE = 'data/cifar-10-batches-bin/test_batch.bin'
# Mean image of the training images (per-channel mean)
MEAN_IMAGE = np.array([125.3069, 122.95015, 113.866]).reshape((3, 1, 1))

def decode_image(image):
    """
    Convert binary data to tensors
    Args:
        image -- Binary data, fixed length
    Return:
        features -- Tensor of shape (None, 3, 32, 32)
        label -- Tensor of shape (None,)
    """
    features = tf.decode_raw(image, tf.uint8)
    features = tf.reshape(features, (3073,))
    label = tf.to_int32(tf.reshape(features[0], ()))
    features = tf.cast(features[1:], tf.float32)
    # Reshape features
    features = tf.reshape(features, (3, 32, 32))
    # Subtract per-channel mean of training images
    features -= MEAN_IMAGE
    return features, label

def train_input_fn(n_shuffle, batch_size, n_repeat):
    """
    Create input_fn for an Estimator
    Args:
        n_shuffle -- Integer, representing shuffle buffer size
        batch_size -- Integer, number of elements in a single batch
        n_repeat -- Integer, representing the number of times this dataset should repeat      
    Return:
        _train_input_fn -- Input function for an Estimator which outputs:
            features -- Tensor of shape (batch_size, 3, 32, 32)
            label -- Tensor of shape (batch_size,)
    """
    def _train_input_fn():
        train_dataset = tf.data.FixedLengthRecordDataset(TRAIN_FILES, 3073).map(decode_image)
        train_dataset = train_dataset.shuffle(n_shuffle).batch(batch_size).repeat(n_repeat)
        features, label = train_dataset.make_one_shot_iterator().get_next()
        return features, label
    return _train_input_fn

def test_input_fn(batch_size):
    """
    Create a dataset for test
    Args:
        batch_size -- Integer, number of elements in a single batch
    Return:
        _test_input_fn -- Input function for an Estimator which outputs:
            features -- Tensor of shape (batch_size, 3, 32, 32)
            label -- Tensor of shape (batch_size,)
    """
    def _test_input_fn():
        test_dataset = tf.data.FixedLengthRecordDataset(TEST_FILE, 3073).map(decode_image)
        test_dataset = test_dataset.batch(batch_size)
        features, label = test_dataset.make_one_shot_iterator().get_next()
        return features, label
    return _test_input_fn

def conv_bn_relu(inputs, filters, kernel_size, padding, mode, params, initializer, name):
    """
    Convinence function for conv2d--batch_normalization--relu layers
    Args:
        inputs -- Tensor, shape of 'NHWC' or 'NCHW'
        filters -- Integer, number of filters
        kernel_size -- Integer or tuple
        padding -- 'SAME' or 'VALID'
        mode -- An instance of tf.estimator.ModeKeys
        params -- Dictionary, passed from tf.estimator.Estimator
        initializer -- Kernel initializer for conv2d
        name -- String
    """
    if params['batch_norm'] == True:
        # Parameters for bn layer
        training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        bn_axis = 1 if params['data_format'] == 'channels_first' else 3
        # Conv layer
        conv = tf.layers.conv2d(inputs,
                                filters,
                                kernel_size,
                                padding=padding,
                                data_format=params['data_format'],
                                kernel_initializer=initializer,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params['reg']),
                                name=name)
        # Batch normalization
        bn = tf.layers.batch_normalization(conv, bn_axis, training, name=name+'_bn')
        return tf.nn.relu(bn, name=name+'_relu')
    else:
        return tf.layers.conv2d(inputs,
                                filters,
                                kernel_size,
                                padding=padding,
                                data_format=params['data_format'],
                                activation=tf.nn.relu,
                                kernel_initializer=initializer,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params['reg']),
                                name=name)
        
def fc_bn_relu(inputs, num_units, mode, params, initializer, name):
    """
    Convinence function for fully_connected--batch_normalization--relu layers
    Args:
        inputs -- Tensor, shape of 'NHWC' or 'NCHW'
        num_units -- Integer, number of units in fully_connected layer
        mode -- tf.estimator.ModeKeys
        params -- Dictionary, passed from tf.estimator.Estimator
        initializer -- Kernel initializer for conv2d
        name -- String
    """
    if params['batch_norm'] == True:
        # Parameters for bn layer
        training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        # Conv layer
        fc = tf.layers.dense(inputs,
                             num_units,
                             kernel_initializer=initializer,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(params['reg']),
                             name=name)
        # Batch normalization
        bn = tf.layers.batch_normalization(fc, -1, training, name=name+'_bn')
        return tf.nn.relu(bn, name=name+'_relu')
    else:
        return tf.layers.dense(inputs,
                               num_units,
                               activation=tf.nn.relu,
                               kernel_initializer=initializer,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(params['reg']),
                               name=name)

