def ppf_basic(args):
    args['port'] = 27007
    args['db_name'] = 'ppf'
    args['col_name'] = 'breakfast'
    args['cache_dir'] = '/data4/shetw/tfutils_cache'
    return args

def batch_size(args):
    args['batch_size'] = 64
    args['test_batch_size'] = 32
    args['fre_filter'] = 10000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 50000000
    return args

def infant_data(args):
    args['col_name'] = 'infant'
    args['data_len'] = 1420
    args['meta_path'] = '/data/shetw/infant_headcam/metafiles/infant_3min_ppf.meta'
    args['frame_root'] = '/data/shetw/infant_headcam/jpgs_extracted'
    args['num_frames'] = 22500 # 15mins in 25fps
    args['file_tmpl'] = "{:06d}.jpg"
    args['flip_frame'] = True
    return args

def ppf_pretrained_vgg():
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)

    args['exp_id'] = 'ppf_test'
    args['from_ckpt'] = '/home/shetw/projects/EventSegmentation/saved_models/vgg_16.ckpt'
    # For testing my reimplementation    
    # args['from_ckpt'] = '/home/shetw/projects/EventSegmentation/saved_models/Zacks_LSTM_AL_S1_1'
    # args['shuffle'] = False
    args['batch_size'] = 40
    return args
    
def ppf_pretrained_vgg_lr8():
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)

    args['exp_id'] = 'ppf_lr8'
    args['from_ckpt'] = '/home/shetw/projects/EventSegmentation/saved_models/vgg_16.ckpt'
    args['batch_size'] = 40
    args['init_lr'] = 1e-8
    return args
    
def ppf_finetune_infant():
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)
    args = infant_data(args)

    args['exp_id'] = 'finetune'
    args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/Zacks_LSTM_AL_S1_1'
    args['batch_size'] = 40
    return args

def ppf_train_infant():
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)
    args = infant_data(args)

    args['exp_id'] = 'train'
    args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/vgg_16.ckpt'
    args['batch_size'] = 40
    return args