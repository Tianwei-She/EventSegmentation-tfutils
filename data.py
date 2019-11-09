from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import tensorflow as tf

def get_feeddict(image, index, prev_emb, prev_state, name_prefix='TRAIN'):
    image_placeholder = tf.get_default_graph().get_tensor_by_name(
            '%s_IMAGE_PLACEHOLDER:0' % name_prefix)
    index_placeholder = tf.get_default_graph().get_tensor_by_name(
            '%s_INDEX_PLACEHOLDER:0' % name_prefix)
    prev_emb_placeholder = tf.get_default_graph().get_tensor_by_name(
            '%s_PREV_EMB_PLACEHOLDER:0' % name_prefix)
    prev_state_placeholder = tf.get_default_graph().get_tensor_by_name(
            '%s_PREV_STATE_PLACEHOLDER:0' % name_prefix)
    feed_dict = {
            image_placeholder: image,
            index_placeholder: index,
            prev_emb_placeholder: prev_emb,
            prev_state_placeholder: prev_state}
    return feed_dict

def get_vgg_feeddict(image, index, name_prefix='TRAIN'):
    image_placeholder = tf.get_default_graph().get_tensor_by_name(
            '%s_IMAGE_PLACEHOLDER:0' % name_prefix)
    index_placeholder = tf.get_default_graph().get_tensor_by_name(
            '%s_INDEX_PLACEHOLDER:0' % name_prefix)
    feed_dict = {
            image_placeholder: image,
            index_placeholder: index}
    return feed_dict


def get_placeholders(batch_size, crop_size=224, num_units=4096, 
                    num_channels=3, name_prefix='TRAIN'):
    image_placeholder = tf.placeholder(
            tf.float32, 
            (batch_size, crop_size, crop_size, num_channels),
            name='%s_IMAGE_PLACEHOLDER' % name_prefix)
    index_placeholder = tf.placeholder(
            tf.int64,
            (batch_size),
            name='%s_INDEX_PLACEHOLDER' % name_prefix)
    prev_emb_placeholder = tf.placeholder(
            tf.float32, 
            (batch_size, num_units), 
            name='%s_PREV_EMB_PLACEHOLDER' % name_prefix)
    prev_state_placeholder = tf.placeholder(
            tf.float32, 
            (batch_size, 2*num_units), 
            name='%s_PREV_STATE_PLACEHOLDER' % name_prefix)
    inputs = {
            'image': image_placeholder,
            'index': index_placeholder,
            'prev_emb': prev_emb_placeholder,
            'prev_state': prev_state_placeholder}
    return inputs
