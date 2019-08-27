
from .core import xavier_lib as ex
import tensorflow as tf


def mobilenet_v2_224(input, train_flag, statistic_flow_builder=None, score_flow_builder=None, gradients_flow_builder=None, EMA=None):
    restore_list = []

    x = ex.conv_bn_block(input, [3, 3], 48, 'Conv', train_flag,
                         restore_list=restore_list,
                         strides=2, moving_average_builder=EMA)
    x = ex.inverted_residual_block(x, 'expanded_conv', train_flag, 24, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder,
                                   expand=False, moving_average_builder=EMA)

    x = ex.inverted_residual_block(x, 'expanded_conv_1', train_flag, 32, 2, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder, moving_average_builder=EMA)
    x = ex.inverted_residual_block(x, 'expanded_conv_2', train_flag, 32, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder,
                                   bypass=True, moving_average_builder=EMA)

    x = ex.inverted_residual_block(x, 'expanded_conv_3', train_flag, 48, 2, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder, moving_average_builder=EMA)
    x = ex.inverted_residual_block(x, 'expanded_conv_4', train_flag, 48, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder,
                                   bypass=True, moving_average_builder=EMA)
    x = ex.inverted_residual_block(x, 'expanded_conv_5', train_flag, 48, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder,
                                   bypass=True, moving_average_builder=EMA)

    x = ex.inverted_residual_block(x, 'expanded_conv_6', train_flag, 88, 2, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder, moving_average_builder=EMA)
    x = ex.inverted_residual_block(x, 'expanded_conv_7', train_flag, 88, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder,
                                   bypass=True, moving_average_builder=EMA)
    x = ex.inverted_residual_block(x, 'expanded_conv_8', train_flag, 88, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder,
                                   bypass=True, moving_average_builder=EMA)
    x = ex.inverted_residual_block(x, 'expanded_conv_9', train_flag, 88, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder,
                                   bypass=True, moving_average_builder=EMA)

    x = ex.inverted_residual_block(x, 'expanded_conv_10', train_flag, 136, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder, moving_average_builder=EMA)
    x = ex.inverted_residual_block(x, 'expanded_conv_11', train_flag, 136, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder,
                                   bypass=True, moving_average_builder=EMA)
    x = ex.inverted_residual_block(x, 'expanded_conv_12', train_flag, 136, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder,
                                   bypass=True, moving_average_builder=EMA)

    x = ex.inverted_residual_block(x, 'expanded_conv_13', train_flag, 224, 2, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder, moving_average_builder=EMA)
    x = ex.inverted_residual_block(x, 'expanded_conv_14', train_flag, 224, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder,
                                   bypass=True, moving_average_builder=EMA)
    x = ex.inverted_residual_block(x, 'expanded_conv_15', train_flag, 224, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder,
                                   bypass=True, moving_average_builder=EMA)

    x = ex.inverted_residual_block(x, 'expanded_conv_16', train_flag, 448, 1, restore_list,
                                   statistic_flow_builder, score_flow_builder, gradients_flow_builder, moving_average_builder=EMA)

    x = ex.conv_bn_block(x, [1, 1], 1792, 'Conv_1', train_flag,
                         statistic_flow_builder, score_flow_builder, gradients_flow_builder,  restore_list, moving_average_builder=EMA)

    with tf.variable_scope('Logits'):
        x = tf.nn.avg_pool(x, [1,7,7,1], [1,1,1,1], 'VALID')
        keep_prob = tf.cond(train_flag, lambda: 0.8, lambda: 1.0)
        x = tf.nn.dropout(x, keep_prob)
        x = ex.convolution(x, [1,1], 1001, 'Conv2d_1c_1x1', True, moving_average_builder=EMA)
        restore_list += [x.convolution_kernels, x.biases]
        x = x.out
        x = tf.squeeze(x)

    return x, restore_list
