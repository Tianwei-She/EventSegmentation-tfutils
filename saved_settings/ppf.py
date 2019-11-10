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
    args['data_len'] = 409
    args['meta_path'] = '/data/shetw/infant_headcam/metafiles/infant_train_ppf.meta'
    args['frame_root'] = '/data/shetw/infant_headcam/jpgs_extracted'
    args['num_frames'] = 22500 # 15mins in 25fps
    args['file_tmpl'] = "{:06d}.jpg"
    args['flip_frame'] = True
    return args

def ppf_pretrained_vgg(): # Node07-1
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
    
def ppf_pretrained_vgg_lr8(): # Node07-2
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)

    args['exp_id'] = 'ppf_lr8'
    args['from_ckpt'] = '/home/shetw/projects/EventSegmentation/saved_models/vgg_16.ckpt'
    args['batch_size'] = 40
    args['init_lr'] = 1e-8
    return args

def ppf_pretrained_vgg_lr6(): # Node08-3
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)

    args['exp_id'] = 'ppf_lr6'
    args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/vgg_16.ckpt'
    args['batch_size'] = 40
    args['init_lr'] = 1e-6
    return args
    
def ppf_pretrained_vgg_lr7():
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)

    args['exp_id'] = 'ppf_lr6'
    args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/vgg_16.ckpt'
    args['batch_size'] = 40
    args['init_lr'] = 1e-7
    return args    

def ppf_train_infant(): # Node08-1
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)
    args = infant_data(args)

    args['exp_id'] = 'train'
    args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/vgg_16.ckpt'
    args['batch_size'] = 40
    return args

def ppf_train_infant_lr7(): # Node08-2
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)
    args = infant_data(args)

    args['exp_id'] = 'train_lr7'
    args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/vgg_16.ckpt'
    args['batch_size'] = 40
    args['init_lr'] = 1e-7
    return args