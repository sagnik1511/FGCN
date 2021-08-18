import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers


def BN(data, bn_momentum = 0.9, name = None):

    return layers.BatchNormalization(momentum = bn_momentum, name = None)(data)

def AC(data, name = None):

    return layers.ReLU(data)

def BN_AC(data, momentum = 0.9, name = None):
    bn = BN(data, bn_momentum = momentum)
    bn_ac = AC(bn)

    return bn_ac

def Conv(data, num_filter, kernel, stride = (1, 1), pad = 'valid', name = None, no_bias = None, w =  None, b = None, attr = None, num_groups = 1):
    if w is None :
        conv = layers.Conv2D(filters = num_filter, kernel_size = kernel, stride = stride, padding = pad, name = name, use_bias = no_bias)(data)

    elif b is None :
        conv = layers.Conv2D(filters = num_filter, kernel_size = kernel, stride = stride, padding = pad, name = name, use_bias = no_bias, kernel_initializer = w)(data)
    else:
        conv = layers.Conv2D(filters = num_filter, kernel_size = kernel, stride = stride, padding = pad, name = name,
                             use_bias = no_bias, kernel_initializer = b)(data)
    return conv

def Conv_BN(data, num_filter, kernel, stride = (1, 1), pad = 'valid', name = None, no_bias = None, w = None, b = None, attr = None, num_groups = 1):
    cov = Conv(data, num_filter, kernel, stride = (1, 1), pad = 'valid', name = None, no_bias = None, w = None, b = None, attr = None, num_groups = 1)
    conv_bn = BN(cov)

    return conv_bn

def Conv_BN_AC(data, num_filter, kernel, stride = (1, 1), pad = 'valid', name = None, no_bias = None, w = None, b = None, attr = None, num_groups = 1):
    conv_bn = Conv_BN(data, num_filter, kernel, stride = (1, 1), pad = 'valid', name = None, no_bias = None, w = None, b = None, attr = None, num_groups = 1)
    conv_bn_ac = AC(conv_bn)

    return conv_bn_ac

def BN_Conv(data, num_filter, kernel, stride = (1, 1), pad = 'valid', name = None, no_bias = None, w = None, b = None, attr = None, num_groups = 1):
    bn = BN(data, num_filter, kernel, stride = (1, 1), pad = 'valid', name = None, no_bias = None, w = None, b = None, attr = None, num_groups = 1)
    bn_conv = Conv(bn, num_filter, kernel, stride= (1, 1), pad = 'valid', name = None, no_bias = None, w = None, b = None, attr = None, num_groups = 1)

    return bn_conv

def AC_Conv(data, num_filter, kernel, stride = (1, 1), pad = 'valid', name = None, no_bias = None, w = None, b = None, attr = None, num_groups = 1):
    ac = AC(data = data)
    ac_conv = Conv(ac, num_filter, kernel, stride = (1, 1), pad = 'valid', name = None, no_bias = None, w = None, b = None, attr = None, num_groups = 1)

    return ac_conv

def BN_AC_Conv(data, num_filter, kernel, stride = (1, 1), pad = 'valid', name = None, no_bias = None, w = None, b = None, attr = None, num_groups = 1):
    bn = BN(data = data)
    bn_ac_conv = AC_Conv(bn, num_filter, kernel, stride = (1, 1), pad = 'valid', name = None, no_bias = None, w = None, b = None, attr = None, num_groups = 1)


    return bn_ac_conv

def Pooling(data, pool_type = 'avg', kernel = (2, 2), pad = 'valid', stride = (2, 2), name = None):
    if pool_type=='avg':
        return layers.AveragePooling2D(pool_size = kernel, stride = stride, padding = pad, name = name)(data)
    else:
        return Layers.MaxPooling2D(pool_size = kernel, stride = stride, padding = pad, name = name)(data)

def ElementWiseSum(x,y,name=None):
    return tf.add(x = x, y = y, name = None)

def Upsampling(lf_data, scale=2,sample_type='nearest',num_args=1,name=None):
    return layers.UpSampling2D(size = (scale, scale), name = None)(lf_data)



