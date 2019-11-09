""" Not in use for now. """
from __future__ import absolute_import, division, print_function

import tensorflow as tf

DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)

class VGG16Model():
    """TODO: An parent/abstract class CNNModel"""
    def __init__(self, emb_size=4096, data_format=None,
                 dtype=DEFAULT_DTYPE):
        if not data_format:
            data_format = ('channels_first' 
                            if tf.test.is_built_with_cuda() 
                            else 'channels_last')
        
        self.emb_size = emb_size
        self.data_format = data_format
        self.dtype = dtype
    
    def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary."""
        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        """Returns a variable scope that the model should be created under."""
        return tf.variable_scope('vgg_16',
                                custom_getter=self._custom_dtype_getter)

    def _preprocess_data(self, inputs):
        if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        return inputs

    def __call__(self, inputs, training):
        """Add operations of a VGG16 network.
        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Set to True to add operations required only when
                training the classifier.

        Returns:
        A logits Tensor with shape [<batch_size>, self.emb_size].
        """
        with self._model_variable_scope() as sc:
            inputs = self._preprocess_data(inputs)
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                                    outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
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
                net = slim.conv2d(net, self.emb_size, [7, 7], padding=fc_conv_padding, scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                                    scope='dropout6')
                net = slim.conv2d(net, self.emb_size, [1, 1], scope='fc7')
                vgg16_Features = tf.reshape(net, (-1,self.emb_size))
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return vgg16_Features

