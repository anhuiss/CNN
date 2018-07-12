"""
Training cifar10 with tf.estimator.Estimator
"""
import tensorflow as tf
import os
import time
from cifar10_utils import train_input_fn, test_input_fn, conv_bn_relu, fc_bn_relu

# Training parameters
N = 50000
N_SHUFFLE = N
EPOCHS_BETWEEN_EVAL = 1

LEARNING_RATE = 1e-3
REGULARIZATION = 0.0
DROPOUT_RATE = 0.5
BATCH_SIZE = 128
BATCH_NORMALIZATION = True
NUM_EPOCHS = 5

MODEL_DIR = 'tmp/cifar10/lr_' + str(LEARNING_RATE) + \
            '_reg_' + str(REGULARIZATION) + '_dropout_' + str(DROPOUT_RATE)
EXPORT_DIR = None
TENSORS_TO_LOG = ['learning_rate', 'cross_entropy', 'training_accuracy']
#EXPORT_DIR = 'tmp/cifar10/v4/saved_model/lr_' + str(LEARNING_RATE) + \
#            '_reg_' + str(REGULARIZATION) + '_dropout_' + str(DROPOUT_RATE)
#TENSORS_TO_LOG = {x:x for x in ['learning_rate', 'cross_entropy', 'training_accuracy']}

# Kernel initializer for conv2d and dense layers
KERNEL_INITILIZER = tf.keras.initializers.he_normal()

def cnn_model(inputs, mode, params):
    """
    Computing logits using tf.layers functional APIs
    Model architure:
        conv(64 filters)--bn--relu--maxpool
        conv(64 filters)--bn--relu--maxpool
        conv(128 filters)--bn--relu--maxpool
        conv(128 filters)--bn--relu--maxpool
        fc--bn--relu--dropout--fc
    Args:
        inputs -- tensor of shape (BATCH_SIZE, 3072)
        mode -- An instance of tf.estimator.ModeKeys
        params -- dictionary, passed from tf.estimator.Estimator
    Return:
        logits -- tensor of shape (BATCH_SIZE, 10)
    """
#    # Reshaping inputs to a 4-D tensor of 'channels_first' format
#    inputs = tf.reshape(inputs, (-1, 3, 32, 32))
    # Reshaping inputs to 'channels_last' format
    if params['data_format'] == 'channels_last':
        inputs = tf.transpose(inputs, (0, 2, 3, 1))
    else:
        assert params['data_format'] == 'channels_first'
        
    # Use conv_bn_relu and fc_bn_relu fast layers to construct model
    # Stage 1
    conv1_1 = conv_bn_relu(inputs, 64, 3, 'SAME', mode, params, KERNEL_INITILIZER, 'conv1_1')
    conv1_2 = conv_bn_relu(conv1_1, 64, 3, 'SAME', mode, params, KERNEL_INITILIZER, 'conv1_2')
    pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2, data_format=params['data_format'], name='pool1')
    
    # Stage 2
    conv2_1 = conv_bn_relu(pool1, 128, 3, 'SAME', mode, params, KERNEL_INITILIZER, 'conv2_1')
    conv2_2 = conv_bn_relu(conv2_1, 128, 3, 'SAME', mode, params, KERNEL_INITILIZER, 'conv2_2')
    pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2, data_format=params['data_format'], name='pool2')
    pool2_flatten = tf.layers.flatten(pool2)
    
    # Stage 4
    fc1 = fc_bn_relu(pool2_flatten, 512, mode, params, KERNEL_INITILIZER, 'fc1')
    if mode == tf.estimator.ModeKeys.TRAIN:
        fc1 = tf.layers.dropout(fc1, rate=params['dropout_rate'], name='dropout_1')
    logits = tf.layers.dense(fc1,
                             10,
                             activation=None,
                             kernel_initializer=KERNEL_INITILIZER,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(params['reg']),
                             name='logits')
    
    return logits

def model_fn(features, labels, mode, params):
    """
    Custom model function to be used by tf.estimator.Estimator
    Args:
        features -- Batch features from input_fn
        labels -- Batch labels from input_fn
        mode -- An instance of tf.estimator.ModeKeys
        params -- Dictionary from tf.estimator.Estimator
    Return:
        Different tf.estimator.EstimatorSpec instances for different mode
    """
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
    # Compute accuracy
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
    if params['batch_norm'] == True:
        # According to tf.layers.batch_normalization API
        updata_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updata_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    else:
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
        classifier.train(input_fn=train_input_fn(N_SHUFFLE, BATCH_SIZE, EPOCHS_BETWEEN_EVAL),
                         hooks=my_hooks)
        
        # Evaluate for test data
        eval_test = classifier.evaluate(input_fn=test_input_fn(BATCH_SIZE))
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
        
    print('\nTime:%.2f\n' % (time.time()-start_time))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

