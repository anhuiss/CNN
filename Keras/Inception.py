"""
Build a simple inception network according to:
    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842
    arXiv:1409.4842 [cs.CV]
"""
import keras
from keras.layers import Conv2D, MaxPool2D, AvgPool2D, Flatten, Dense, Dropout

def inception_block(X, filters, stage, initializer='he_normal'):
    """
    Origin version inception block
    Args:
        X -- Input tensor, shape: (batch_size, W, H, C)
        filters -- List of number filters, corresponding to
            1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, pool_proj convolutions
        stage -- Integer, used to name the stages
        initializer -- Kernel initializer
    Return:
        Output tensor
    """
    f1, f3_r, f3, f5_r, f5, fp = filters
    basename = 'stage' + str(stage) + '_'
    
    X_1 = Conv2D(f1, kernel_size=3, padding='SAME', activation='relu',
                 kernel_initializer=initializer, name=basename+'1_1')(X)
    X_3_reduce = Conv2D(f3_r, kernel_size=3, padding='SAME', activation='relu',
                        kernel_initializer=initializer, name=basename+'3_3_reduce')(X)
    X_3 = Conv2D(f3, kernel_size=3, padding='SAME', activation='relu',
                 kernel_initializer=initializer, name=basename+'3_3')(X)
    X_5_reduce = Conv2D(f5_r, kernel_size=3, padding='SAME', activation='relu',
                        kernel_initializer=initializer, name=basename+'5_5_reduce')(X)
    X_5 = Conv2D(f5, kernel_size=3, padding='SAME', activation='relu',
                 kernel_initializer=initializer, name=basename+'5_5')(X)
    X_pooling = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='SAME',
                          name=basename+'incep_pool')(X)
    X_p = Conv2D(fp, kernel_size=3, padding='SAME', activation='relu',
                 kernel_initializer=initializer, name=basename+'pool_1_1')(X)
    
    output = keras.layers.concatenate([X_1, X_3, X_5, X_p], axis=-1)
    return output

def inception_v1(input_shape, n_classes, initializer='he_normal'):
    X_input = keras.Input(shape=input_shape)
    
    # Stage 1
    X = Conv2D(32, kernel_size=(3, 3), padding='SAME', activation='relu',
               kernel_initializer=initializer, name='stage1_conv1')(X_input)
    X = Conv2D(32, kernel_size=(3, 3), padding='SAME', activation='relu',
               kernel_initializer=initializer, name='stage1_conv2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='stage1_maxpool')(X)
    # Stage 2
    X = inception_block(X, [64, 96, 128, 16, 32, 32], stage=2, initializer=initializer)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='stage2_maxpool')(X)
    # Stage 3
    X = inception_block(X, [128, 128, 192, 32, 96, 64], stage=3, initializer=initializer)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='stage3_maxpool')(X)
    # Stage 4
    X = AvgPool2D(pool_size=(4, 4))(X)
    X = Flatten()(X)
    X = Dropout(0.5)(X)
    X = Dense(n_classes, activation='softmax',
              kernel_initializer=initializer, name='stage4_dense1')(X)
    
    model = keras.Model(inputs=X_input, outputs=X)
    
    return model