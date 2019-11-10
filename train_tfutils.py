from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import tensorflow as tf
import collections
import argparse

from tfutils import base, optimizer
import tfutils.defaults

import config
from models import main_model
import data
from dataset import FrameDataset

import pdb

BREAKFAST_DATA_LEN = 1989
# TRAIN_WINDOW_SIZE = 5


def get_config():
    """TODO: Modify config.py"""
    cfg = config.Config()
    cfg.add('exp_id', type=str, required=True,
            help='Name of experiment ID')
    cfg.add('batch_size', type=int, default=32,
            help='Training batch size')
    cfg.add('test_batch_size', type=int, default=16,
            help='Testing batch size')
    cfg.add('gpu', type=str, required=True,
            help='Value for CUDA_VISIBLE_DEVICES')
    # Not sure what this is used for
    cfg.add('gpu_offset', type=int, default=0,
            help='GPU offset, useful for KMeans?')
    cfg.add('data_len', type=int, default=BREAKFAST_DATA_LEN,
            help='Total number of videos in the input dataset')
    cfg.add('pure_test', type=bool, default=False,
            help='Whether just testing.')
    
    # Model
    cfg.add('model_type', type=str, default='vgg_16',
            help='vgg_16')
    cfg.add('emb_size', type=int, default=4096,
            help='Size of the embedding')
    cfg.add('num_units', type=int, default=4096,
            help='Number of units in LSTM')
    cfg.add('train_window_size', type=int, default=5,
            help='Window size for adaptive learning')
    
    
    # Data
    cfg.add('meta_path', type=str, default='/data4/shetw/breakfast/metafiles/videos_train_split1.meta',
            help='Path to metafile')
    cfg.add('frame_root', type=str, default='/data4/shetw/breakfast/extracted_frames',
            help='Root path to frames')
    cfg.add('num_frames', type=int, default=1500,
            help='Number of frames fed into the LSTM')
    cfg.add('crop_size', type=int, default=224,
            help='Size of the cropped input')
    cfg.add('flip_frame', type=bool, default=False,
            help='Whether to flip frames or not (for infant videos)')
    cfg.add('file_tmpl', type=str, default="Frame_{:06d}.jpg",
            help='Size of the cropped input')
    cfg.add('shuffle', type=bool, default=True,
            help='Shuffle the dataset or not during training')
    
    # Saving parameters
    cfg.add('port', type=int, required=True,
            help='Port number for mongodb')
    cfg.add('host', type=str, default='localhost',
            help='Host for mongodb')
    cfg.add('db_name', type=str, required=True,
            help='Name of database')
    cfg.add('col_name', type=str, required=True,
            help='Name of collection')
    cfg.add('cache_dir', type=str, required=True,
            help='Prefix of cache directory for tfutils')
    cfg.add('fre_valid', type=int, default=10009,
            help='Frequency of validation')
    cfg.add('fre_metric', type=int, default=1000,
            help='Frequency of saving metrics')
    cfg.add('fre_filter', type=int, default=10009,
            help='Frequency of saving filters')
    cfg.add('fre_cache_filter', type=int,
            help='Frequency of caching filters')
    
    # Loading parameters
    cfg.add('load_exp', type=str, default=None,
            help='The experiment to load from, in the format '
                 '[dbname]/[collname]/[exp_id]')
    cfg.add('load_port', type=int,
            help='Port number of mongodb for loading (defaults to saving port')
    cfg.add('load_step', type=int,
            help='Step number for loading')
    cfg.add('resume', type=bool,
            help='Flag for loading from last step of this exp_id, will override'
            ' all other loading options.')
    cfg.add('from_ckpt', type=str, default=None,
            help='The ckpt file path to be loaded from')
    
    # Learning rate
    cfg.add('init_lr', type=float, default=5e-9,
            help='Initial learning rate')
    cfg.add('big_lr', type=float, default=1e-8,
            help='Bigger learning rate in adaptive training')
    cfg.add('small_lr', type=float, default=1e-9,
            help='Smaller learning rate in adaptive training')
    cfg.add('target_lr', type=float, default=None,
            help='Target leraning rate for ramping up')
    cfg.add('lr_boundaries', type=str, default=None,
            help='Learning rate boundaries for 10x drops')
    cfg.add('ramp_up_epoch', type=int, default=1,
            help='Number of epoch for ramping up')

    return cfg

def get_save_load_params_from_arg(args):
    # save_params: defining where to save the models
    args.fre_cache_filter = args.fre_cache_filter or args.fre_filter
    cache_dir = os.path.join(
            args.cache_dir, '.tfutils', 'localhost:%i' % args.port,
            args.db_name, args.col_name, args.exp_id)
    save_params = {
            'host': 'localhost',
            'port': args.port,
            'dbname': args.db_name,
            'collname': args.col_name,
            'exp_id': args.exp_id,
            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': args.fre_metric,
            'save_valid_freq': args.fre_valid,
            'save_filters_freq': args.fre_filter,
            'cache_filters_freq': args.fre_cache_filter,
            'cache_dir': cache_dir,
            }

    # load_params: defining where to load, if needed
    load_port = args.load_port or args.port
    load_dbname = args.db_name
    load_collname = args.col_name
    load_exp_id = args.exp_id
    load_query = None
    from_ckpt = None

    if not args.resume:
        if args.load_exp is not None:
            load_dbname, load_collname, load_exp_id = args.load_exp.split('/')
        if args.load_step:
            load_query = {'exp_id': load_exp_id,
                          'saved_filters': True,
                          'step': args.load_step}
            print('Load query', load_query)
        if args.from_ckpt is not None:
            from_ckpt = args.from_ckpt

    load_params = {
            'host': 'localhost',
            'port': load_port,
            'dbname': load_dbname,
            'collname': load_collname,
            'exp_id': load_exp_id,
            'do_restore': True,
            'query': load_query,
            'from_ckpt': from_ckpt,
            }
    return save_params, load_params


def get_model_func_params(args):
    model_params = {
        'model_type': args.model_type,
        'emb_size': args.emb_size,
        'num_units': args.num_units,
    }
    return model_params

def get_train_data_param_from_arg(args):
    if args.model_type == 'vgg_16':
        train_data_param = {
            'func': data.get_placeholders,
            'batch_size': args.batch_size,
            }
    return train_data_param

def loss_func(output, *args, **kwargs):
    return output['loss']
"""
def get_lr_from_boundary_and_ramp_up(
        global_step, boundaries, 
        init_lr, target_lr, ramp_up_epoch,
        NUM_BATCHES_PER_EPOCH):
    curr_epoch  = tf.div(
            tf.cast(global_step, tf.float32), 
            tf.cast(NUM_BATCHES_PER_EPOCH, tf.float32))
    curr_phase = (tf.minimum(curr_epoch/float(ramp_up_epoch), 1))
    curr_lr = init_lr + (target_lr-init_lr) * curr_phase

    if boundaries is not None:
        boundaries = boundaries.split(',')
        boundaries = [int(each_boundary) for each_boundary in boundaries]

        all_lrs = [
                curr_lr * (0.1 ** drop_level) \
                for drop_level in range(len(boundaries) + 1)]

        curr_lr = tf.train.piecewise_constant(
                x=global_step,
                boundaries=boundaries, values=all_lrs)
    return curr_lr
"""

# online_loss = collections.deque(maxlen=TRAIN_WINDOW_SIZE) # Keep the running average of the losses
def get_adaptive_learning_rate(global_step, init_lr, big_lr, small_lr):
    """ TODO: Implement adaptive learning """
    """ TODO: Check if the learning rate assignment is done after the loss is updated"""
    """
    curr_loss = tf.get_default_graph().get_tensor_by_name('mse_loss:0')
    if len(online_loss) == 0:
        return tf.constant(init_lr, dtype=tf.float32)
    else:
        avg_loss = np.mean(online_loss)
        online_loss.append(curr_loss)
        if curr_loss > avg_loss:
            print(big_lr, curr_loss, avg_loss, online_loss)
            return tf.constant(big_lr, dtype=tf.float32)
        else:
            print(small_lr, curr_loss, avg_loss, online_loss)
            return tf.constant(small_lr, dtype=tf.float32)
    """
    return tf.constant(init_lr, dtype=tf.float32)


def get_loss_lr_opt_params_from_arg(args):

    loss_params = {
        'pred_targets': [],
        'loss_func': loss_func,
    }

    # learning_rate_params: build the learning rate
    # For now, just stay the same
    """ TODO: Need to change to adaptive learning """
    learning_rate_params = {
            'func': get_adaptive_learning_rate,
            'init_lr': args.init_lr,
            'small_lr': args.small_lr,
            'big_lr': args.big_lr,
            }

    # optimizer_params: use tfutils optimizer,
    # as mini batch is implemented there
    optimizer_params = {
            'optimizer': tf.train.GradientDescentOptimizer,
            }
    return loss_params, learning_rate_params, optimizer_params


def get_params_from_arg(args):
    
    # === Save & load params ===
    save_params, load_params = get_save_load_params_from_arg(args)

    # === Model params ===
    model_func_params = get_model_func_params(args)
    vgg_emb_node, lstm_state_node, loss_node = [], [], []
    def build_output(inputs, train, **kwargs):
        # res = {'loss': sseLoss, 'lstm_state': new_state, 'vgg_emb': vgg_emb}
        res = main_model.build_output(inputs, train, **model_func_params)
        outputs, logged_cfg, _vgg_emb, _lstm_state, _loss = res
        vgg_emb_node.append(_vgg_emb)
        lstm_state_node.append(_lstm_state)
        loss_node.append(_loss)
        return outputs, logged_cfg

    model_params = {'func': build_output}
    multi_gpu = len(args.gpu.split(',')) - args.gpu_offset
    if multi_gpu > 1:
        model_params['num_gpus'] = multi_gpu
        model_params['devices'] = ['/gpu:%i' \
                                   % (idx + args.gpu_offset) \
                                   for idx in range(multi_gpu)]

    # === Train params ===
    if args.pure_test:
        train_params, loss_params = {}, {}
        learning_rate_params, optimizer_params = {}, {}
    else:
        # Data enumerator 
        train_frame_dataset = FrameDataset(args.frame_root, args.meta_path, \
                                     args.batch_size, args.num_frames,
                                     file_tmpl=args.file_tmpl, shuffle=args.shuffle)
        num_steps_per_epoch = train_frame_dataset.num_batch_per_epoch * args.num_frames
        train_frame_generator = train_frame_dataset.batch_of_frames_generator()
        train_frame_enumerator = [enumerate(train_frame_generator)]
        
        # Data params (defining input placeholders)
        train_data_param = get_train_data_param_from_arg(args)
        
        prev_emb_np, prev_state_np = [], []
        # Train_loop
        def train_loop(sess, train_targets, num_minibatches=1, **params):
            global_step_vars = [v for v in tf.global_variables() \
                                if 'global_step' in v.name]
            assert len(global_step_vars) == 1
            global_step = sess.run(global_step_vars[0])

            # Update the data_loader for each epoch
            if global_step % num_steps_per_epoch==0 and global_step!=0:
                train_frame_enumerator.pop()
                train_frame_enumerator.append(enumerate(train_frame_generator))
            
            # Initialization of prev_emb & prev_state 
            # at the beginning of each batch
            if global_step % args.num_frames == 0:
                _, (image, index, step) = train_frame_enumerator[0].next()
                assert step == 0
                assert len(prev_state_np) <= 1
                if len(prev_state_np) == 1:
                    prev_state_np.pop()
                
                np.random.seed(6) # Test my reimplementation
                prev_state_np.append(np.random.uniform(low=-0.5, high=0.5, \
                                    size=(args.batch_size, 2*args.num_units)))
                
                assert len(prev_emb_np) <= 1
                if len(prev_emb_np) == 1:
                    prev_emb_np.pop()
                vgg_feed_dict = data.get_vgg_feeddict(image, index)
                prev_emb_np.append(sess.run(vgg_emb_node[0], feed_dict=vgg_feed_dict))
            
            # Normal train step  
            # Get data from the enumerator
            _, (image, index, step) = train_frame_enumerator[0].next()
            assert global_step % args.num_frames + 1 == step
            # Feed input data and run
            # TODO: Learning rate for adaptive learning 
            feed_dict = data.get_feeddict(image, index, \
                                          prev_emb_np[0], prev_state_np[0])
            sess_res = sess.run(train_targets+loss_node+vgg_emb_node+lstm_state_node, feed_dict=feed_dict)
            _, vgg_emb, lstm_state = sess_res[-3], sess_res[-2], sess_res[-1] # _ is the pred errors [bs]
            sess_res = [sess_res[0]]
            prev_emb_np[0], prev_state_np[0] = vgg_emb, lstm_state   
            return sess_res
            
        train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': None,
            'thres_loss': float('Inf'),
            'num_steps': float('Inf'),
            'train_loop': {'func': train_loop},
        }
    
        # === Loss, learning_rate & optimizer params ===
        loss_params, learning_rate_params, optimizer_params \
            = get_loss_lr_opt_params_from_arg(args)

    
    # === Validation params ===
    """ TODO """  
    validation_params = {}


    params = {
            'save_params': save_params,
            'load_params': load_params,
            'model_params': model_params,
            'train_params': train_params,
            'loss_params': loss_params,
            'learning_rate_params': learning_rate_params,
            'optimizer_params': optimizer_params,
            'log_device_placement': False,
            'validation_params': validation_params,
            'skip_check': True,
            }
    return params

def main():
    # Parse arguments
    cfg = get_config()
    args = cfg.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Get params needed, start training
    params = get_params_from_arg(args)
    if not args.pure_test:
        base.train_from_params(**params)
    else:
        params.pop('learning_rate_params')
        params.pop('optimizer_params')
        params.pop('loss_params')
        params.pop('train_params')
        base.test_from_params(**params)


if __name__ == "__main__":
    main()