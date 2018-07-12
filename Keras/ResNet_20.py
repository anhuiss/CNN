"""
Build a ResNet_20 for cifar10 according to:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
"""
import keras
from keras.layers import Conv2D, BatchNormalization, Dense, Activation, Add
from keras.layers import Flatten, AveragePooling2D
from keras.regularizers import l2

def identity_block(X, filters, data_format, stage, block, initializer, regularizer):
    """
    Building an identity block "skipping over" 2 hidden layers with the following structure:
        mainpath: conv(filters, kernel_size=3, strides=1, 'SAME') -- bn -- relu
                  conv(filters, kernel_size=3, strides=1, 'SAME') -- bn
        shortcut: X_shortcut = X
    Args:
        X -- Input tensor with shape (N, C, H, W) if data_format is 'channels_first'
             or (N, H, W, C) if  data_format is 'channels_last'
        filters -- List of integers indicating number of filters in each conv layer
        data_format -- 'channels_first' or 'channels_last'
        stage -- Integer, used to name the layers, depending on the position in the network
        block -- String/character, used to name the layers, depending on the position
                 in the network
        initializer -- String or keras.initializer function
        regularizer -- keras.regularizer function
    Return:
        A tensor of same shape as X
    """
    X_shortcut = X
    
    # Determine batch normalization axis
    bn_axis = 1 if data_format == 'channels_first' else 3    
    
    # Defining name basis
    conv_base_name = 'res' + str(stage) + block + '_branch'
    bn_base_name = 'bn' + str(stage) + block + '_branch'
    
    # First component of main path
    X = Conv2D(filters=filters, kernel_size=3, strides=1, padding='SAME',
               data_format=data_format,
               kernel_initializer=initializer,
               kernel_regularizer=regularizer,
               name=conv_base_name+'2a')(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name+'2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters=filters, kernel_size=3, strides=1, padding='SAME',
               data_format=data_format,
               kernel_initializer=initializer,
               kernel_regularizer=regularizer,
               name=conv_base_name+'2b')(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name+'2b')(X)
    
    # Add shortcut to main path and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def conv_block(X, filters, data_format, stage, block, initializer, regularizer):
    """
    Building an identity block "skipping over" 2 hidden layers with the following structure:
        mainpath: conv(filters, kernel_size=3, strides=2, 'SAME') -- bn -- relu
                  conv(filters, kernel_size=3, strides=1, 'SAME') -- bn
        shortcut: conv(filters, kernel_size=1, strides=2, 'SAME') -- bn
    Args:
        X -- Input tensor with shape (N, C, H, W) if data_format is 'channels_first'
             or (N, H, W, C) if  data_format is 'channels_last'
        filters -- List of integers indicating number of filters in each conv layer
        data_format -- 'channels_first' or 'channels_last'
        stage -- Integer, used to name the layers, depending on the position in the network
        block -- String/character, used to name the layers, depending on the position
                 in the network
        initializer -- String or keras.initializer function
        regularizer -- keras.regularizer function
    Return:
        A tensor of same shape as X
    """
    X_shortcut = X
    
    # Determine batch normalization axis
    bn_axis = 1 if data_format == 'channels_first' else 3    
    
    # Defining name basis
    conv_base_name = 'res' + str(stage) + block + '_branch'
    bn_base_name = 'bn' + str(stage) + block + '_branch'
    
    # First component of main path
    X = Conv2D(filters=filters, kernel_size=3, strides=2, padding='SAME',
               data_format=data_format,
               kernel_initializer=initializer,
               kernel_regularizer=regularizer,
               name=conv_base_name+'2a')(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name+'2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters=filters, kernel_size=3, strides=1, padding='SAME',
               data_format=data_format,
               kernel_initializer=initializer,
               kernel_regularizer=regularizer,
               name=conv_base_name+'2b')(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name+'2b')(X)
    
    # Shortcut path:
    X_shortcut = Conv2D(filters=filters, kernel_size=1, strides=2, padding='SAME',
                        data_format=data_format,
                        kernel_initializer=initializer,
                        kernel_regularizer=regularizer,
                        name=conv_base_name+'1')(X_shortcut)
    X_shortcut = BatchNormalization(axis=bn_axis, name=bn_base_name+'1')(X_shortcut)
    
    # Add shortcut to main path and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def ResNet_20(input_shape, n_classes, data_format, initializer='he_normal',
              regularizer=l2(1e-3)):
    """
    Build a ResNet_20 model
    Model structure:
        CONV-BN-RELU
        IDBLOCK(filters=16)*3
        IDBLOCK(filters=32)*3
        IDBLOCK(filters=64)*3
        AVGPOOL
        FLATTEN
        FC
    Inputs:
        input_shape -- Shape without batch_size, needed for the input layer
        n_classes -- Integer, number of classes
        data_format -- 'channels_first' or 'channels_last'
        initializer -- String or keras.initializer function
        regularizer -- keras.regularizer function
    Return:
        A Model() instance in keras
    """
    # Input for keras.Model
    X_input = keras.Input(shape=input_shape)
    
    # Determine batch normalization axis
    bn_axis = 1 if data_format == 'channels_first' else 3    
    
    # Stage 1
    X = Conv2D(filters=16, kernel_size=3, strides=1, padding='SAME',
               data_format=data_format,
               kernel_initializer=initializer,
               kernel_regularizer=regularizer,
               name='conv1')(X_input)
    X = BatchNormalization(axis=bn_axis, name='conv1_bn')(X)
    X = Activation('relu')(X)
    
    # Stage 2
    X = identity_block(X, filters=16, data_format=data_format, stage=2, block='a',
                       initializer=initializer, regularizer=regularizer)
    X = identity_block(X, filters=16, data_format=data_format, stage=2, block='b',
                       initializer=initializer, regularizer=regularizer)
    X = identity_block(X, filters=16, data_format=data_format, stage=2, block='c',
                       initializer=initializer, regularizer=regularizer)
    
    # Stage 3
    X = conv_block(X, filters=32, data_format=data_format, stage=3, block='a',
                   initializer=initializer, regularizer=regularizer)
    X = identity_block(X, filters=32, data_format=data_format, stage=3, block='b',
                       initializer=initializer, regularizer=regularizer)
    X = identity_block(X, filters=32, data_format=data_format, stage=3, block='c',
                       initializer=initializer, regularizer=regularizer)
    
    # Stage 4
    X = conv_block(X, filters=64, data_format=data_format, stage=4, block='a',
                   initializer=initializer, regularizer=regularizer)
    X = identity_block(X, filters=64, data_format=data_format, stage=4, block='b',
                       initializer=initializer, regularizer=regularizer)
    X = identity_block(X, filters=64, data_format=data_format, stage=4, block='c',
                       initializer=initializer, regularizer=regularizer)
    
    # Stage 5
    X = AveragePooling2D(pool_size=2, strides=2, padding='VALID',
                         data_format=data_format, name='avg_pool')(X)
    X = Flatten()(X)
    X = Dense(units=n_classes, activation='softmax',
              kernel_initializer=initializer,
              kernel_regularizer=regularizer,
              name='fc')(X)
    
    model = keras.Model(inputs=X_input, outputs=X, name='ResNet_20')
    
    return model