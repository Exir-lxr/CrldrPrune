import tensorflow as tf

# CONFIG
INITIAL_STDDEV = 0.09


def analyze_name(var_name):
    part_list = []
    tmp = ''
    for i in var_name:
        if i == '/':
            part_list.append(tmp)
            tmp = ''
        else:
            tmp += i
    part_list.append(tmp[:-2])
    return part_list


class exponential_moving_average(object):
    '''
    Only be used on Variable not Tensor!
    '''
    def __init__(self, variable):
        self.var = variable
        self.decay_rate_maps_update_op_dict = {}
        with tf.variable_scope(analyze_name(variable.name)[-1]):
            self.avg = tf.get_variable('ExponentialMovingAverage', shape=variable.get_shape().as_list(),
                                       initializer=tf.zeros_initializer, dtype=tf.float32, trainable=False)
        self.assign_avg_to_var = tf.assign(self.var, self.avg)
        self.assign_var_to_avg = tf.assign(self.avg, self.var)

    def set_avg_value_op(self):
        return self.assign_var_to_avg

    def set_var_value_op(self):
        return self.assign_avg_to_var

    def get_update_op(self, decay_rate=0.9999):
        if not (decay_rate in self.decay_rate_maps_update_op_dict):
            self.decay_rate_maps_update_op_dict[decay_rate] =tf.assign(
                self.avg, tf.add(tf.multiply(decay_rate, self.avg), tf.multiply(1.0 - decay_rate, self.var)))
        return self.decay_rate_maps_update_op_dict[decay_rate]


class exponential_moving_average_builder(object):

    def __init__(self, decay_rate=0.9999):
        self.decay_rate = decay_rate
        self.var_list = []
        self.moving_avg_list = []
        self.update_avg_ops = []
        self.set_avg_to_var_ops = []
        self.set_var_to_avg_ops = []

    def collect_variable(self, var):
        self.var_list.append(var)
        with tf.variable_scope(analyze_name(var.name)[-1]):
            avg = tf.get_variable('ExponentialMovingAverage', shape=var.get_shape().as_list(),
                            initializer=tf.zeros_initializer, dtype=tf.float32, trainable=False)

        self.moving_avg_list.append(avg)
        self.update_avg_ops.append(
            tf.assign(avg,
                      tf.add(tf.multiply(self.decay_rate, avg),tf.multiply(1.0 - self.decay_rate, var))))
        self.set_avg_to_var_ops.append(tf.assign(var, avg))
        self.set_var_to_avg_ops.append(tf.assign(avg, var))
        return True


class convolution_return(object):
    def __init__(self):
        self.convolution_kernels = None
        self.biases = None
        self.out = None


def convolution(previous_layer, kernel_shape, depth, layer_name=None,
                biases=False, strides=1, trainable=True, weights_decay=0.00004, padding='SAME',
                weights_name='weights', biases_name='biases', moving_average_builder=None, validate_shape=True):
    if layer_name is None:
        a = convolution_return()

        in_shape = previous_layer.get_shape()[3]

        a.convolution_kernels = tf.get_variable(weights_name,
                                                [int(kernel_shape[0]), int(kernel_shape[1]), in_shape, depth],
                                                tf.float32,
                                                tf.truncated_normal_initializer(stddev=INITIAL_STDDEV),
                                                tf.contrib.layers.l2_regularizer(weights_decay),
                                                trainable=trainable,
                                                validate_shape=validate_shape
                                                )

        a.out = tf.nn.conv2d(previous_layer, a.convolution_kernels, strides=[1, strides, strides, 1], padding=padding)

        if moving_average_builder is not None:
            moving_average_builder.collect_variable(a.convolution_kernels)

        if biases:
            a.biases = tf.get_variable(biases_name, [depth], tf.float32, tf.zeros_initializer, trainable=trainable,
                                       validate_shape=validate_shape)
            if moving_average_builder is not None:
                moving_average_builder.collect_variable(a.biases)
            a.out = tf.add(a.out, a.biases)
    else:
        with tf.variable_scope(layer_name):
            a = convolution_return()

            in_shape = previous_layer.get_shape()[3]

            a.convolution_kernels = tf.get_variable(weights_name,
                                                    [int(kernel_shape[0]), int(kernel_shape[1]), in_shape, depth],
                                                    tf.float32,
                                                    tf.truncated_normal_initializer(stddev=INITIAL_STDDEV),
                                                    tf.contrib.layers.l2_regularizer(weights_decay),
                                                    trainable=trainable,
                                                    validate_shape=validate_shape
                                                    )

            a.out = tf.nn.conv2d(previous_layer, a.convolution_kernels, strides=[1, strides, strides, 1], padding=padding)

            if moving_average_builder is not None:
                moving_average_builder.collect_variable(a.convolution_kernels)

            if biases:
                a.biases = tf.get_variable(biases_name, [depth], tf.float32, tf.zeros_initializer, trainable=trainable,
                                           validate_shape=validate_shape)
                if moving_average_builder is not None:
                    moving_average_builder.collect_variable(a.biases)
                a.out = tf.add(a.out, a.biases)
    return a


def depthwise_convolution(previous_layer, kernel_shape, channel_multiplier, layer_name=None,
                          biases=False, strides=1, trainable=True, weights_decay=0.00004, padding='SAME',
                          weights_name='depthwise_weights', biases_name='biases', moving_average_builder=None,
                           validate_shape=True):
    if layer_name is None:
        a = convolution_return()

        in_shape = previous_layer.get_shape()[3]

        a.convolution_kernels = tf.get_variable(weights_name,
                                                [int(kernel_shape[0]), int(kernel_shape[1]), in_shape,
                                                 channel_multiplier],
                                                tf.float32,
                                                tf.truncated_normal_initializer(stddev=INITIAL_STDDEV),
                                                tf.contrib.layers.l2_regularizer(weights_decay),
                                                trainable=trainable,
                                                validate_shape=validate_shape
                                                )

        a.out = tf.nn.depthwise_conv2d(previous_layer, a.convolution_kernels, [1, strides, strides, 1], padding)

        if moving_average_builder is not None:
            moving_average_builder.collect_variable(a.convolution_kernels)

        if biases:
            a.biases = tf.get_variable(biases_name, [channel_multiplier * in_shape], tf.float32,
                                       tf.zeros_initializer,
                                       trainable=trainable, validate_shape=validate_shape)
            if moving_average_builder is not None:
                moving_average_builder.collect_variable(a.biases)
            a.out = tf.add(a.out, a.biases)
    else:
        with tf.variable_scope(layer_name):
            a = convolution_return()

            in_shape = previous_layer.get_shape()[3]

            a.convolution_kernels = tf.get_variable(weights_name,
                                                    [int(kernel_shape[0]), int(kernel_shape[1]), in_shape,channel_multiplier],
                                                    tf.float32,
                                                    tf.truncated_normal_initializer(stddev=INITIAL_STDDEV),
                                                    tf.contrib.layers.l2_regularizer(weights_decay),
                                                    trainable=trainable,
                                                    validate_shape=validate_shape
                                                    )

            a.out = tf.nn.depthwise_conv2d(previous_layer, a.convolution_kernels, [1, strides, strides, 1], padding)

            if moving_average_builder is not None:
                moving_average_builder.collect_variable(a.convolution_kernels)

            if biases:
                a.biases = tf.get_variable(biases_name, [channel_multiplier * in_shape], tf.float32, tf.zeros_initializer,
                                               trainable=trainable, validate_shape=validate_shape)
                if moving_average_builder is not None:
                    moving_average_builder.collect_variable(a.biases)
                a.out = tf.add(a.out, a.biases)
    return a


class moving_average_return(object):
    def __init__(self):
        self.value = None
        self.op_list = []


def moving_average(variable, avg_name, decay=0.99, zero_stay=False, validate_shape=True):
    '''
    :param variable:    A tensor, the variable to compute the moving average.
    :param avg_name:    A string, the name.
    :param decay:       A float, decay rate.
    :param zero_stay:   Bool, whether ignore the situation where variable becomes 0.
    :return:            value:      A tensor, record the value after moving.
                        op_list:    A list of op tensor, run which to update the moving value.
                        ...
    '''

    a = moving_average_return()

    if zero_stay:
        decay = 1 - (1-decay)*tf.abs(tf.sign(variable))
    a.value = tf.get_variable(avg_name, shape=variable.get_shape().as_list(),
                              initializer=tf.zeros_initializer, dtype=tf.float32, trainable=False,
                              validate_shape=validate_shape)
    a.op_list.append(tf.assign(a.value, tf.add(tf.multiply(decay, a.value), tf.multiply(1.0 - decay, variable))))
    return a


class batch_normalization_return(object):
    def __init__(self):
        self.mean_avg = None
        self.mean = None
        self.var_avg = None
        self.var = None
        self.beta = None
        self.gamma = None
        self.out = None
        self.op_list = []


def batch_normalization(layers, train_flag, layer_name, epsilon,
                        trainable=True, decay=0.997, zero_stay=False,
                        beta_name='beta', gamma_name='gamma',
                        moving_mean_name='moving_mean', moving_var_name='moving_variance',
                        moving_average_builder=None, validate_shape=True):
    a = batch_normalization_return()

    with tf.variable_scope(layer_name):
        if len(layers.shape) == 4:
            depth = layers.shape[3]
            a.mean, a.var = tf.nn.moments(layers, [0, 1, 2])
        else:
            depth = layers.shape[1]
            a.mean, a.var = tf.nn.moments(layers, [0])
        a.beta = tf.get_variable(beta_name, initializer=tf.zeros_initializer, shape=[depth],
                                 dtype=tf.float32, trainable=trainable, validate_shape=validate_shape)
        a.gamma = tf.get_variable(gamma_name, initializer=tf.ones_initializer, shape=[depth],
                                  dtype=tf.float32, trainable=trainable, validate_shape=validate_shape)
        mm_cls = moving_average(a.mean, moving_mean_name, decay, zero_stay=zero_stay, validate_shape=validate_shape)
        mv_cls = moving_average(a.var, moving_var_name, decay, zero_stay=zero_stay, validate_shape=validate_shape)
        a.op_list += mm_cls.op_list
        a.mean_avg = mm_cls.value
        a.op_list += mv_cls.op_list
        a.var_avg = mv_cls.value
        m, v = tf.cond(train_flag, lambda: [a.mean, a.var], lambda: [a.mean_avg, a.var_avg])
        if zero_stay:
            a.out = tf.multiply(tf.abs(tf.sign(layers)),
                                tf.nn.batch_normalization(layers, m, v, a.beta, a.gamma, epsilon))
        else:
            a.out = tf.nn.batch_normalization(layers, m, v, a.beta, a.gamma, epsilon)
        if moving_average_builder is not None:
            moving_average_builder.collect_variable(a.beta)
            moving_average_builder.collect_variable(a.gamma)
    return a


import os
print(os.getcwd())
from . import crelindr_lib as crldr


# OLD VERSION: specialized for pruning on mobilenet_v2
def conv_bn_block(previous_layer, kernel_shape, depth, block_name,
                  train_flag,
                  statistic_flow_builder=None, score_flow_builder=None, gradients_flow_builder=None,
                  restore_list=None,
                  strides=1, padding='SAME', activate_fn='relu6', moving_average_builder=None):
    with tf.variable_scope(block_name):
        in_shape = previous_layer.get_shape()[3]

        if statistic_flow_builder is not None:
            input_mask = tf.get_variable('input_mask', in_shape, tf.float32, tf.ones_initializer, trainable=False)
            previous_layer = tf.multiply(previous_layer, input_mask)

            sr = statistic_flow_builder.build_statistic_flow_on_conv_inputs(previous_layer, kernel_shape, strides, padding)
            var_after, stacked_trans = crldr.variances_after_decorrelation_and_corresponding_transformations(
                sr.covariance, sr.zero_var_mask_non_zero, sr.zero_var_mask_zero)

            a = convolution(previous_layer,kernel_shape, depth, None, strides=strides,padding=padding, moving_average_builder=moving_average_builder)

            if restore_list is not None:
                restore_list.append(a.convolution_kernels)

            score_flow_builder.collect_weights_mask(a.convolution_kernels, input_mask, sr.zero_var_mask_one)
            score_flow_builder.collect_variances_trans(var_after, stacked_trans)

            b = batch_normalization(a.out, train_flag, 'BatchNorm', 1e-3, moving_average_builder=moving_average_builder)

            if restore_list is not None:
                restore_list += [b.beta, b.gamma, b.mean_avg, b.var_avg]

            tf.add_to_collection('bn_update_op_list', b.op_list[0])
            tf.add_to_collection('bn_update_op_list', b.op_list[1])

            gradients_flow_builder.collect_xs(b.out, kernel_shape, strides, padding)

        else:
            a = convolution(previous_layer, kernel_shape, depth, None, strides=strides, padding=padding,
                            moving_average_builder=moving_average_builder)

            if restore_list is not None:
                restore_list.append(a.convolution_kernels)

            b = batch_normalization(a.out, train_flag, 'BatchNorm', 1e-3, moving_average_builder=moving_average_builder)

            if restore_list is not None:
                restore_list += [b.beta, b.gamma, b.mean_avg, b.var_avg]

            tf.add_to_collection('bn_update_op_list', b.op_list[0])
            tf.add_to_collection('bn_update_op_list', b.op_list[1])

        if activate_fn == 'relu6':
            return tf.nn.relu6(b.out)
        elif activate_fn == 'relu':
            return tf.nn.relu(b.out)
        else:
            return b.out


def depthwise_conv_bn_block(previous_layer, train_flag, restore_list=None,
                            kernel_shape=[3, 3], block_name='depthwise', channel_multiplier=1,
                            strides=1, padding='SAME', activate_fn='relu6', moving_average_builder=None):
    with tf.variable_scope(block_name):

        a = depthwise_convolution(previous_layer, kernel_shape, channel_multiplier, None, strides=strides, padding=padding,
                                  moving_average_builder=moving_average_builder)

        if restore_list is not None:
            restore_list.append(a.convolution_kernels)

        b = batch_normalization(a.out, train_flag, 'BatchNorm', 1e-3, moving_average_builder=moving_average_builder)

        if restore_list is not None:
            restore_list += [b.beta, b.gamma, b.mean_avg, b.var_avg]

        tf.add_to_collection('bn_update_op_list', b.op_list[0])
        tf.add_to_collection('bn_update_op_list', b.op_list[1])

        if activate_fn == 'relu6':
            return tf.nn.relu6(b.out)
        elif activate_fn == 'relu':
            return tf.nn.relu(b.out)
        else:
            return b.out


def inverted_residual_block(previous_layer, block_name, train_flag, out_depth, strides, restore_list=None,
                            statistic_flow_builder=None, score_flow_builder=None, gradients_flow_builder=None,
                            expand_rate=6, expand=True, bypass=False, moving_average_builder=None):
    with tf.variable_scope(block_name):
        in_shape = previous_layer.get_shape().as_list()[3]
        if expand:
            x = conv_bn_block(previous_layer, [1, 1], expand_rate*in_shape, 'expand', train_flag,
                              statistic_flow_builder, score_flow_builder, gradients_flow_builder, restore_list,
                              moving_average_builder=moving_average_builder)
        else:
            x = previous_layer

        x = depthwise_conv_bn_block(x, train_flag, restore_list, strides=strides,
                                    moving_average_builder=moving_average_builder)

        x = conv_bn_block(x, [1, 1], out_depth, 'project', train_flag,
                          statistic_flow_builder, score_flow_builder, gradients_flow_builder, restore_list,
                          activate_fn='None', moving_average_builder=moving_average_builder)

        if bypass:
            if in_shape != out_depth:
                print('ERROR: Channel Number Error, cannot build bypass.')
            else:
                x = tf.add(x, previous_layer)

        return x
