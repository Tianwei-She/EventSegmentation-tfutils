from __future__ import division, print_function, absolute_import
import os, sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

def vgg_16(input_frame, is_training, emb_size):
    with tf.variable_scope('vgg_16', reuse=tf.AUTO_REUSE) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                                outputs_collections=end_points_collection):
            net = slim.repeat(input_frame, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, emb_size, [7, 7], padding='VALID', scope='fc6')
            net = slim.dropout(net, 0.8, is_training=is_training,
                                                scope='dropout6')
            net = slim.conv2d(net, emb_size, [1, 1], scope='fc7')
            vgg_emb = tf.reshape(net, (-1, emb_size)) 
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    return vgg_emb, end_points

def lstm_cell(W, b, forget_bias, inputs, state):
	one = constant_op.constant(1, dtype=dtypes.int32)
	add = math_ops.add
	multiply = math_ops.multiply
	sigmoid = math_ops.sigmoid
	activation = math_ops.sigmoid
	# activation = math_ops.tanh

	c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

	gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), W)
	gate_inputs = nn_ops.bias_add(gate_inputs, b)
	# i = input_gate, j = new_input, f = forget_gate, o = output_gate
	i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)

	forget_bias_tensor = constant_op.constant(forget_bias, dtype=f.dtype)

	new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), activation(j)))
	new_h = multiply(activation(new_c), sigmoid(o))
	new_state = array_ops.concat([new_c, new_h], 1)

	return new_h, new_state

def build_output(inputs, train, 
                model_type='vgg_16',
                emb_size=4096,
                num_units=4096,
                **kwargs):
    """
    Inputs:
        input_frame = tf.placeholder(tf.float32, (None, 224, 224, 3), name='input_frame')
        prev_emb = tf.placeholder(tf.float32, (None, num_units), name='prev_emb')
        prev_state = tf.placeholder(tf.float32, (None, 2*num_units), name="prev_state")
    """
    input_frame, prev_emb, prev_state = inputs['image'], inputs['prev_emb'], inputs['prev_state']
    is_training = train
    is_training = False
    # This will be stored in the db
    logged_cfg = {'kwargs': kwargs}

    # VGG16
    if model_type == 'vgg_16':
        vgg_emb, _ = vgg_16(input_frame, is_training, emb_size)
            
    # LSTM
    W_lstm = vs.get_variable("W1", shape=[emb_size + num_units, 4*num_units])
    b_lstm = vs.get_variable("b1", shape=[4*num_units], initializer=init_ops.zeros_initializer(dtype=tf.float32))
    pred_emb, new_state = lstm_cell(W_lstm, b_lstm, 1.0, prev_emb, prev_state)
    
    # Loss
    sseLoss1 = tf.square(tf.subtract(vgg_emb, pred_emb))
    sseLoss = tf.reduce_mean(sseLoss1, axis=1)

    # Loss value node for adaptive learning
    # loss_var = tf.Variable(0, name='mse_loss', trainable=False, dtype=tf.float32)
    # loss_assign = tf.assign(loss_var, sseLoss)
    
    ret_dict = {
        'loss': sseLoss,
        'index': inputs['index'],
    }
    return ret_dict, logged_cfg, vgg_emb, new_state, sseLoss # , loss_assign