# -*- coding: utf-8 -*-
###############################################################################
# Batch Normalization, Regularization, Dropout
###############################################################################
import tensorflow as tf
import numpy as np
import os
import time

N = 50000
EPOCHS_BETWEEN_EVAL = 1

LEARNING_RATE = 1e-4
REGULARIZATION = 0.0
DROPOUT_RATE = 0.0
BATCH_SIZE = 128
BATCH_NORMALIZATION = False
NUM_EPOCHS = 5

TRAIN_FILES = ['data/cifar-10-batches-bin/data_batch_' + str(i) + '.bin' for i in range(1, 6)]
TEST_FILE = 'data/cifar-10-batches-bin/test_batch.bin'
MODEL_DIR = 'tmp/cifar10/v4/lr_' + str(LEARNING_RATE) + \
            '_reg_' + str(REGULARIZATION) + '_dropout_' + str(DROPOUT_RATE)
EXPORT_DIR = None
#EXPORT_DIR = 'tmp/cifar10/v4/saved_model/lr_' + str(LEARNING_RATE) + \
#            '_reg_' + str(REGULARIZATION) + '_dropout_' + str(DROPOUT_RATE)
#TENSORS_TO_LOG = {x:x for x in ['learning_rate', 'cross_entropy', 'training_accuracy']}
TENSORS_TO_LOG = ['learning_rate', 'cross_entropy', 'training_accuracy']
# Mean image of the training images (per-channel mean)
MEAN_IMAGE = np.array([125.3069, 122.95015, 113.866]).reshape((3, 1, 1))
# Transform MEAN_IMAGE into shape (3072,)
#T_MEAN_IMAEG = np.concatenate([np.full((1024,),
#                                       MEAN_IMAGE[i],
#                                       dtype=np.float32) for i in range(3)])

KERNEL_INITILIZER = tf.keras.initializers.he_normal()

def decode_image(image):
    """
    Convert binary data to features and label
    Args:
        image -- binary data, fixed length
    Return:
        features -- tensor of shape (3072,), value of this tensor is in [0, 1]
        label -- tensor of shape ()
    """
    features = tf.decode_raw(image, tf.uint8)
    features = tf.reshape(features, (3073,))
    label = tf.to_int32(tf.reshape(features[0], ()))
    features = tf.cast(features[1:], tf.float32)
    # Reshape features
    features = tf.reshape(features, (3, 32, 32))
    # Subtract per-channel mean
    features -= MEAN_IMAGE
    return features, label

def train_input_fn():
    train_dataset = tf.data.FixedLengthRecordDataset(TRAIN_FILES, 3073).map(decode_image)
    train_dataset = train_dataset.shuffle(N).batch(BATCH_SIZE).repeat(EPOCHS_BETWEEN_EVAL)
    features, label = train_dataset.make_one_shot_iterator().get_next()
    
    return features, label

def test_input_fn():
    test_dataset = tf.data.FixedLengthRecordDataset(TEST_FILE, 3073).map(decode_image)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    features, label = test_dataset.make_one_shot_iterator().get_next()
    
    return features, label

def conv_bn_relu(inputs, filters, kernel_size, padding, mode, params, name):
#    initializer = tf.zeros_initializer()
#    initializer = tf.random_normal_initializer(stddev=1e-4)
#    initializer = tf.keras.initializers.he_normal()
    
    if params['batch_norm'] == True:
        # For tf.layers.batch_normalization
        training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        bn_axis = 1 if params['data_format'] == 'channels_first' else 3
        # Conv layer
        conv = tf.layers.conv2d(inputs,
                                filters,
                                kernel_size,
                                padding=padding,
                                data_format=params['data_format'],
                                kernel_initializer=KERNEL_INITILIZER,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params['reg']),
                                name=name)
        # Batch_norm layer
        bn = tf.layers.batch_normalization(conv, bn_axis, training, name=name+'_bn')
        return tf.nn.relu(bn, name=name+'_relu')
    else:
        return tf.layers.conv2d(inputs,
                                filters,
                                kernel_size,
                                padding=padding,
                                data_format=params['data_format'],
                                activation=tf.nn.relu,
                                kernel_initializer=KERNEL_INITILIZER,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params['reg']),
                                name=name)
        
def fc_bn_relu(inputs, num_units, mode, params, name):
#    initializer = tf.zeros_initializer()
#    initializer = tf.random_normal_initializer(stddev=1e-4)
#    initializer = tf.keras.initializers.he_normal()
    
    if params['batch_norm'] == True:
        # For tf.layers.batch_normalization
        training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        # Conv layer
        fc = tf.layers.dense(inputs,
                             num_units,
                             kernel_initializer=KERNEL_INITILIZER,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(params['reg']),
                             name=name)
        # Batch_norm layer
        bn = tf.layers.batch_normalization(fc, -1, training, name=name+'_bn')
        return tf.nn.relu(bn, name=name+'_relu')
    else:
        return tf.layers.dense(inputs,
                               num_units,
                               activation=tf.nn.relu,
                               kernel_initializer=KERNEL_INITILIZER,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(params['reg']),
                               name=name)

def cnn_model(inputs,mode, params):
    """
    Computing logits using tf.layers 'function API'
    Args:
        inputs -- tensor of shape (BATCH_SIZE, 3072)
    Return:
        logits -- tensor of shape (BATCH_SIZE,)
    """
    # Reshaping inputs to a 4-D tensor of 'channels_first' format
    inputs = tf.reshape(inputs, (-1, 3, 32, 32))
    # Reshaping inputs to 'channels_last' format
    if params['data_format'] == 'channels_last':
        inputs = tf.transpose(inputs, (0, 2, 3, 1))
    else:
        assert params['data_format'] == 'channels_first'
        
    # Use conv_bn_relu and fc_bn_relu to construct model
    conv1_1 = conv_bn_relu(inputs, 64, 3, 'SAME', mode, params, 'conv1_1')
    conv1_2 = conv_bn_relu(conv1_1, 64, 3, 'SAME', mode, params, 'conv1_2')
    pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2,
                                    data_format=params['data_format'], name='pool1')
    
    conv2_1 = conv_bn_relu(pool1, 128, 3, 'SAME', mode, params, 'conv2_1')
    conv2_2 = conv_bn_relu(conv2_1, 128, 3, 'SAME', mode, params, 'conv2_2')
    pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2,
                                    data_format=params['data_format'], name='pool2')
    
    conv3_1 = conv_bn_relu(pool2, 256, 3, 'SAME', mode, params, 'conv3_1')
    conv3_2 = conv_bn_relu(conv3_1, 256, 3, 'SAME', mode, params, 'conv3_2')
    pool3 = tf.layers.max_pooling2d(conv3_2, 2, 2,
                                    data_format=params['data_format'], name='pool3')
    pool3_flatten = tf.layers.flatten(pool3)
    
    fc1 = fc_bn_relu(pool3_flatten, 1024, mode, params, 'fc1')
    if mode == tf.estimator.ModeKeys.TRAIN:
        fc1 = tf.layers.dropout(fc1, rate=params['dropout_rate'], name='dropout_1')
    fc2 = fc_bn_relu(fc1, 1024, mode, params, 'fc2')
    if mode == tf.estimator.ModeKeys.TRAIN:
        fc2 = tf.layers.dropout(fc2, rate=params['dropout_rate'], name='dropout_2')
    logits = tf.layers.dense(fc2,
                             10,
                             activation=None,
                             kernel_initializer=KERNEL_INITILIZER,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(params['reg']),
                             name='logits')
    
    return logits

###################################################################################      
#    # Regularization
#    reg = params['reg']
#    
#    # Building cnn architecture using functional APIs
#    conv1_1 = tf.layers.conv2d(inputs, 64, 3, padding='SAME',
#                               data_format=params['data_format'],
#                               activation=tf.nn.relu,
#                               kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
#                               name='conv1_1')
#    conv1_2 = tf.layers.conv2d(conv1_1, 64, 3, padding='SAME',
#                               data_format=params['data_format'],
#                               activation=tf.nn.relu,
#                               kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
#                               name='conv1_2')
#    pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2,
#                                    data_format=params['data_format'], name='pool1')
#    conv2_1 = tf.layers.conv2d(pool1, 128, 3, padding='SAME',
#                               data_format=params['data_format'],
#                               activation=tf.nn.relu,
#                               kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
#                               name='conv2_1')
#    conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, padding='SAME',
#                               data_format=params['data_format'],
#                               activation=tf.nn.relu,
#                               kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
#                               name='conv2_2')
#    pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2,
#                                    data_format=params['data_format'], name='pool2')
#    conv3_1 = tf.layers.conv2d(pool2, 256, 3, padding='SAME',
#                               data_format=params['data_format'],
#                               activation=tf.nn.relu,
#                               kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
#                               name='conv3_1')
#    conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, padding='SAME',
#                               data_format=params['data_format'],
#                               activation=tf.nn.relu,
#                               kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
#                               name='conv3_2')
#    pool3 = tf.layers.max_pooling2d(conv3_2, 2, 2,
#                                    data_format=params['data_format'], name='pool3')
#    pool3_flatten = tf.layers.flatten(pool3)
#    fc1 = tf.layers.dense(pool3_flatten,
#                          units=1024,
#                          activation=tf.nn.relu,
#                          kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
#                          name='fc1')
#    if mode == tf.estimator.ModeKeys.TRAIN:
#        fc1 = tf.layers.dropout(fc1, rate=params['dropout_rate'], name='dropout')
#    fc2 = tf.layers.dense(fc1,
#                          units=1024,
#                          activation=tf.nn.relu,
#                          kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
#                          name='fc2')
#    if mode == tf.estimator.ModeKeys.TRAIN:
#        fc2 = tf.layers.dropout(fc2, rate=params['dropout_rate'], name='dropout')
#    logits = tf.layers.dense(fc2,
#                             10,
#                             activation=None,
#                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
#                             name='logits')
#    return logits
###################################################################################   


def model_fn(features, labels, mode, params):
    # For CLI serving (see serving_input_receiver_fn in main())
    if isinstance(features, dict):
        features = features['image']
        
    # Compute logits
    logits = cnn_model(features, mode, params)
    
    # Output for PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        predicted_classes = tf.argmax(logits, 1)
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(
                mode,
                predictions=predictions,
                export_outputs={
                        'classify': tf.estimator.export.PredictOutput(predictions)
                })
        
    # Compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Add regularization
    if params['reg'] != 0.0:
        loss += tf.losses.get_regularization_loss()
    # Computer accuracy
    accuracy = tf.metrics.accuracy(labels, tf.argmax(logits, 1))
    # Eval metrics
    eval_metrics = {'accuracy': accuracy}
    
    # Output for EVAL mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metrics)
    
    assert mode == tf.estimator.ModeKeys.TRAIN
    
    # Tensors to log
    tf.identity(params['learning_rate'], 'learning_rate')
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy[1], 'training_accuracy')
    
    # Add to tensorboard
    tf.summary.scalar('accuracy', accuracy[1])

    # Create an optimizer and a train_op
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    # According to tf.layers.batch_normalization API
    updata_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(updata_ops):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op) 
    
def main(argv):
    start_time = time.time()
    
    # Determine data_format
    data_format = ('channels_first' 
                   if tf.test.is_built_with_cuda else 'channels_last')
    
    # Check MODEL_DIR
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Create a classifier
    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        params={
                                            'learning_rate': LEARNING_RATE,
                                            'reg': REGULARIZATION,
                                            'dropout_rate': DROPOUT_RATE,
                                            'batch_norm': BATCH_NORMALIZATION,
                                            'data_format': data_format
                                        },
                                        model_dir=MODEL_DIR)
    
    for i in range(NUM_EPOCHS):
        # Create hooks
        my_hooks = [tf.train.LoggingTensorHook(TENSORS_TO_LOG,
                                               every_n_iter=100)]
        
        # Train 'EPOCHS_BETWEEN_EVAL' epochs
        classifier.train(input_fn=train_input_fn, hooks=my_hooks)
        
        # Evaluate for test data
        eval_test = classifier.evaluate(input_fn=test_input_fn)
        print('\nEvaluation results:\n%s\n' % eval_test)
        
    # Export the model
    if EXPORT_DIR is not None:
        # Make a directory to save the model
        if not os.path.exists(EXPORT_DIR):
            os.makedirs(EXPORT_DIR)
        # Create serving_input_receiver_fn
        image = tf.placeholder(tf.float32, (None, 28, 28))
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
                {'image': image}
            )
        # Export
        classifier.export_savedmodel(EXPORT_DIR, input_fn)
        
    end_time = time.time()
    print('\nTime:%.2f\n' % (start_time-end_time))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

