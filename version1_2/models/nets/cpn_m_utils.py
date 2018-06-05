# -*- coding: utf-8 -*-
import tensorflow as tf
import pickle
import tensorflow.contrib.slim as slim
import sys, os
import numpy as np
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.framework import ops   #tensorflow/tensorflow/python/ops/ ?
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import regularizers, \
    initializers, layers
from tensorflow.contrib.layers.python.layers import utils

BN_EPSILON = 0.001

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def batch_normalization_layer(input_layer, dimension,name):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    with tf.variable_scope(name):
        beta = tf.get_variable(name='beta',
                                shape = dimension, 
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0, tf.float32),
                                regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002))
        gamma = tf.get_variable(name='gamma',
                                shape = dimension, 
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0, tf.float32),
                                regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer

def subsample(inputs, factor, name=None):
    """Subsamples the input along the spatial dimensions.

    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.
      scope: Optional variable_scope.

    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
      return inputs
    else:
      return layers.max_pool2d(inputs, [1, 1], stride=factor, scope=name)

def conv_bn(inputs,
            filters,
            kernel_size,
            stride,
            activation,        
            name):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs=inputs,
                                filters=filters,
                                kernel_size=[kernel_size, kernel_size],
                                strides=[stride, stride],
                                padding='same',
                                activation=activation,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='conv')
        bn = batch_normalization_layer(conv, filters,name='bn')
    return bn
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               name,
               rate=1):
    with tf.variable_scope(name):
        depth_in = inputs.get_shape().as_list()[-1]
        conv1 = tf.layers.conv2d(inputs=inputs,
                                filters=depth_bottleneck,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                dilation_rate=(rate,rate),
                                padding='same',
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='conv1')
        bn1 = batch_normalization_layer(conv1, depth_bottleneck,name='bn1')

        conv2 = tf.layers.conv2d(inputs=tf.nn.relu(bn1),
                                filters=depth_bottleneck,
                                kernel_size=[3, 3],
                                strides=[stride, stride],
                                dilation_rate=(rate,rate),
                                padding='same',
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='conv2')
        bn2 = batch_normalization_layer(conv2, depth_bottleneck,name='bn2')
    
        conv3 = tf.layers.conv2d(inputs=tf.nn.relu(bn2),
                                filters=depth,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                dilation_rate=(rate,rate),
                                padding='same',
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='conv3')
        bn3 = batch_normalization_layer(conv3, depth,name='bn3')
        if depth == depth_in:
            shortcut = subsample(inputs, factor=stride, name='shortcut')
        else:
            shortcut = tf.layers.conv2d(inputs=inputs,
                                filters=depth,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                padding='same',
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name=None)
            shortcut = subsample(shortcut, factor=stride, name='shortcut')
        shortcut = batch_normalization_layer(shortcut, depth,name='shortcut')
    
        output =tf.nn.relu(bn3 + shortcut,name='output')
    return output
