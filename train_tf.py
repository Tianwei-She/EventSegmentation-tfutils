from __future__ import division, print_function, absolute_import
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
import collections
import numpy as np
import tensorflow as tf
from random import sample
from PIL import Image
from tqdm import tqdm

from tensorflow.contrib import slim

import pdb


meta_path = "/data4/shetw/breakfast/metafiles/videos_train_split1.meta"
frame_root = "/data4/shetw/breakfast/extracted_frames"
batch_size = 64

"""
meta_path = "/data4/shetw/breakfast/metafiles/test_videos_metafile.txt"
frame_root = "/data4/shetw/breakfast/test_extracted_frames"
batch_size = 1
"""

PURE_TEST = False
train = False
ADAPTIVE_LEARNING = False

num_epochs = 5
num_frames = 1500
IMAGENET_MEAN = np.array([123.68, 116.779, 103.939])

# Data
def read_from_metafile(meta_path):
    with open(meta_path, 'r') as f:
        lines = f.readlines()
        video_list = [(line.split()[0], int(line.split()[1])) for line in lines]
        print(video_list[0])
    return video_list

def load_batches(video_list, batch_size):
    # shuffled_video_list = sample(video_list, len(video_list))
    batches = []
    for i in range(len(video_list)//batch_size):
        # batches.append(shuffled_video_list[i*batch_size:(i+1)*batch_size])
        batches.append(video_list[i*batch_size:(i+1)*batch_size])
    return batches

def get_frames_at_step(batch, step):
    """ Returns the step-th resized & normalized frame of each video in the batch
    Args:
        batch: a list of video information
        step: starts from 0
    Returns:
        An array in size [batch_size, height, width, 3]
    """
    frame_batch_list = []
    for video in batch:
        # pdb.set_trace()
        vd_name, vd_length = video
        real_step = step % vd_length
        frame_path = os.path.join(frame_root, vd_name, 
                                    "Frame_{:06d}.jpg".format(real_step+1))
        image = Image.open(frame_path)
        image = image.resize((224, 224), Image.ANTIALIAS)
        image = np.array(image)
        image = np.subtract(image, IMAGENET_MEAN)
        frame_batch_list.append(image)
    frame_batch = np.stack(frame_batch_list)
    return frame_batch

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

# ====== Data =======
video_list = read_from_metafile(meta_path)

# ====== Model ======
num_units = 4096
emb_size = 4096

input_frame = tf.placeholder(tf.float32, (None, 224, 224, 3), name='input_frame')
prev_emb = tf.placeholder(tf.float32, (None, num_units), name='prev_emb')

# VGG-16
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
        net = slim.dropout(net, 0.8, is_training=train,
                                            scope='dropout6')
        net = slim.conv2d(net, emb_size, [1, 1], scope='fc7')
        vgg_emb = tf.reshape(net, (-1, emb_size)) 
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

"""
# The input to vgg.vgg_16 should be a tensor instead of a placeholder
with slim.arg_scope(vgg.vgg_arg_scope()):
    vgg_emb, _ = vgg.vgg_16(input_frame, num_classes=None, is_training=True)
    # Check the shape of the VGG output
    pdb.set_trace()
"""

# LSTM
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

prev_state = tf.placeholder(tf.float32, [None, 2*num_units], name="state")
W_lstm = vs.get_variable("W1", shape=[emb_size + num_units, 4*num_units])
b_lstm = vs.get_variable("b1", shape=[4*num_units], initializer=init_ops.zeros_initializer(dtype=tf.float32))

# pred_emb, new_state = lstm_cell(W_lstm, b_lstm, 1.0, vgg_emb, prev_state)
pred_emb, new_state = lstm_cell(W_lstm, b_lstm, 1.0, prev_emb, prev_state)

"""
# Use LSTM cell will cause error
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
# pred_emb, new_state = lstm(prev_emb, state=prev_state)
pred_emb, new_state = tf.nn.static_rnn(lstm_cell, inputs=[prev_emb], initial_state=prev_state)
print("LSTM state size:{}".format(lstm_cell.state_size))
print("Predicted feature size:{}".format(pred_emb.shape))
"""

# Loss
sseLoss1 = tf.square(tf.subtract(vgg_emb, pred_emb))
# mask = tf.greater(sseLoss1, learnError * tf.ones_like(sseLoss1))
# sseLoss1 = tf.multiply(sseLoss1, tf.cast(mask, tf.float32))
sseLoss = tf.reduce_mean(sseLoss1, axis=1)
# return sseLoss, vgg_emb, new_state

# ====== Train ======
# def train(video_list, loss, vgg_emb, new_state):
"""
Args:
    video_list: a list of dicts of video information
"""
lr = 1e-5
learning_rate = tf.placeholder(tf.float32, [])
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(sseLoss)

init = tf.global_variables_initializer()
vgg_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16"))
global_saver = tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
    sess.run(init)
    # vgg_saver.restore(sess, "../EventSegmentation/saved_models/vgg_16.ckpt")
    global_saver.restore(sess, '../EventSegmentation/saved_models/Zacks_LSTM_AL_S1_1')
    if PURE_TEST:
        videos = load_batches(video_list, 1)
        for video_i, video in tqdm(enumerate(videos)):
            print(video)
            error_file_path = os.path.join('error_files', video[0][0]+'.txt')
            error_file = open(error_file_path, 'a')

            np.random.seed(6)
            prev_state_np = np.random.uniform(low=-0.5, high=0.5, \
                            size=(batch_size, 2*num_units))
            prev_frame_np = get_frames_at_step(video, 0)
            prev_emb_np = sess.run(vgg_emb, feed_dict={input_frame: prev_frame_np})

            # import pdb; pdb.set_trace()
            for frame in tqdm(range(num_frames-1)):
                curr_frame = get_frames_at_step(video, frame+1)
                loss, new_emb_np, new_state_np = sess.run([sseLoss, vgg_emb, new_state], \
                                                    feed_dict={
                                                        input_frame: curr_frame,
                                                        prev_emb: prev_emb_np,
                                                        prev_state: prev_state_np,
                                                        learning_rate: lr
                                                    })
                pdb.set_trace()
                prev_emb_np = new_emb_np
                prev_state_np = new_state_np
                print("Batch {} - Frame {}: Loss {}".format(video_i, frame+1, np.mean(loss)))
                error_file.write('%d\t%2f\n'%(frame, loss))

    for epoch in range(num_epochs):
        print("====== Epoch {} ======".format(epoch))
        batches = load_batches(video_list, batch_size)
        losses = []
        for batch_i, batch in tqdm(enumerate(batches)):
            # Batch is a list of video information (vd_name, vd_length)
            # Read in the first frame
            np.random.seed(6)
            prev_state_np = np.random.uniform(low=-0.5, high=0.5, \
                            size=(batch_size, 2*num_units))
            prev_frame_np = get_frames_at_step(batch, 0)
            prev_emb_np = sess.run(vgg_emb, feed_dict={input_frame: prev_frame_np})

            predError = collections.deque(maxlen=30)
            avgPredError = 1

            for frame in tqdm(range(num_frames-1)):
                curr_frame = get_frames_at_step(batch, frame+1)
                import pdb; pdb.set_trace()
                _, loss, new_emb_np, new_state_np = sess.run([train_op, sseLoss, vgg_emb, new_state], \
                                                    feed_dict={
                                                        input_frame: curr_frame,
                                                        prev_emb: prev_emb_np,
                                                        prev_state: prev_state_np,
                                                        learning_rate: lr
                                                    })
                pdb.set_trace()
                prev_emb_np = new_emb_np
                prev_state_np = new_state_np
                print("Batch {} - Frame {}: Loss {}".format(batch_i, frame+1, np.mean(loss)))

                if ADAPTIVE_LEARNING:
                    if np.mean(loss)/avgPredError > 1.5:
					 	lr = 1e-5
						print('Gating n_steps=', frame, avgPredError, np.mean(loss))
						# predError.clear()
                    else:
                        lr = 1e-7
						# print('NOT Gating n_steps=', segCount, avgPredError, ret[1])
					 	
                predError.append(np.mean(loss))
                avgPredError = np.mean(predError)
                
            global_saver.save(sess, 'saved_models/PPF-TF-lr6-Epoch{}-Batch'.format(epoch), global_step=batch_i)