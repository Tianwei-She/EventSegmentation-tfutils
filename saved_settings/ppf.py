def ppf_basic(args):
    args['port'] = 27007
    args['db_name'] = 'ppf'
    args['col_name'] = 'breakfast'
    args['cache_dir'] = '/data4/shetw/tfutils_cache'
    args['num_frames'] = 1500 # Breakfast
    return args

def batch_size(args):
    args['batch_size'] = 64
    args['test_batch_size'] = 32
    args['fre_filter'] = 10000
    args['fre_cache_filter'] = 5000
    args['fre_valid'] = 50000000 # No validation during training
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

def alice_data(args):
    args['col_name'] = 'alice'
    args['data_len'] = 338
    args['meta_path'] = '/data4/shetw/infant_headcam/metafiles/infant_alice.meta'
    args['frame_root'] = '/data4/shetw/infant_headcam/jpgs_extracted'
    args['num_frames'] = 37500 # 25 mins in 25fps
    args['file_tmpl'] = "{:06d}.jpg"
    args['flip_frame'] = True
    return args

def sam_data(args):
    args['col_name'] = 'sam'
    args['data_len'] = 903
    args['meta_path'] = '/data4/shetw/infant_headcam/metafiles/infant_sam.meta'
    args['frame_root'] = '/data4/shetw/infant_headcam/jpgs_extracted'
    args['num_frames'] = 22500 # 15 mins in 25fps
    args['file_tmpl'] = "{:06d}.jpg"
    args['flip_frame'] = True
    return args

def ppf_test_random():
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)

    args['col_name'] = 'breakfast_loss'
    args['exp_id'] = 'ppf_test_sample_ramdom'
    args['pure_test'] = True
    args['from_ckpt'] = '/home/shetw/projects/EventSegmentation/saved_models/vgg_16.ckpt'
    args['test_meta_path'] = '/data4/shetw/breakfast/metafiles/videos_test_split1_sample.meta'
    return args

def ppf_test_lr6(): # Node07-1
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)

    args['col_name'] = 'breakfast_loss'
    args['exp_id'] = 'ppf_test_sample_lr6'
    args['load_exp'] = 'ppf/breakfast/ppf_lr6'
    args['pure_test'] = True
    args['test_meta_path'] = '/data4/shetw/breakfast/metafiles/videos_test_split1_sample.meta'
    return args

def ppf_test_lr7(): # Node07-4
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)

    args['col_name'] = 'breakfast_loss'
    args['exp_id'] = 'ppf_test_sample_lr7'
    args['load_exp'] = 'ppf/breakfast/ppf_lr7'
    args['pure_test'] = True
    args['test_meta_path'] = '/data4/shetw/breakfast/metafiles/videos_test_split1_sample.meta'
    return args

def ppf_test_lr59(): # Node07-4
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)

    args['col_name'] = 'breakfast_loss'
    args['exp_id'] = 'ppf_test_sample_lr59'
    args['load_exp'] = 'ppf/breakfast/ppf_test'
    args['pure_test'] = True
    args['test_meta_path'] = '/data4/shetw/breakfast/metafiles/videos_test_split1_sample.meta'
    return args

def ppf_pretrained_vgg(): # Node07-1
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)

    args['exp_id'] = 'ppf_test'
    # args['from_ckpt'] = '/home/shetw/projects/EventSegmentation/saved_models/vgg_16.ckpt'
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
    # args['from_ckpt'] = '/home/shetw/projects/EventSegmentation/saved_models/vgg_16.ckpt'
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
    
def ppf_pretrained_vgg_lr7(): # Node07-4 - saving ckpt error at 51000 step
    args = {}
    args = ppf_basic(args)
    args = batch_size(args)

    args['exp_id'] = 'ppf_lr7'
    # args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/vgg_16.ckpt'
    args['batch_size'] = 128
    args['init_lr'] = 1e-7
    # Test multi-gpu
    # args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/Zacks_LSTM_AL_S1_1'
    # args['shuffle'] = False
    return args    

def ppf_train_infant(): # Node08-1
    args = {}
    args = ppf_basic(args)
    args = infant_data(args)
    args = batch_size(args)

    args['exp_id'] = 'train'
    args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/vgg_16.ckpt'
    args['batch_size'] = 40
    return args

def ppf_train_infant_lr7(): # Node08-2
    args = {}
    args = ppf_basic(args)
    args = infant_data(args)
    args = batch_size(args)

    args['exp_id'] = 'train_lr7'
    args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/vgg_16.ckpt'
    args['batch_size'] = 40
    args['init_lr'] = 1e-7
    return args

def ppf_train_infant_lr58(): # Node07-3
    args = {}
    args = ppf_basic(args)
    args = infant_data(args)
    args = batch_size(args)

    args['exp_id'] = 'train_lr58'
    args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/vgg_16.ckpt'
    args['meta_path'] = '/data4/shetw/infant_headcam/metafiles/infant_train_ppf.meta'
    args['frame_root'] = '/data4/shetw/infant_headcam/jpgs_extracted'
    args['batch_size'] = 128
    args['init_lr'] = 5e-8
    return args

def ppf_train_alice(): # Node07-1
    args = {}
    args = ppf_basic(args)
    args = alice_data(args)
    args = batch_size(args)

    args['exp_id'] = 'train_lr7'
    args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/vgg_16.ckpt'
    args['pure_train'] = True
    args['batch_size'] = 128
    args['init_lr'] = 1e-7
    return args

def ppf_train_sam(): 
    args = {}
    args = ppf_basic(args)
    args = sam_data(args)
    args = batch_size(args)

    args['exp_id'] = 'train_lr7'
    args['from_ckpt'] = '/data4/shetw/breakfast/saved_models/vgg_16.ckpt'
    args['pure_train'] = True
    args['batch_size'] = 128
    args['init_lr'] = 1e-7
    return args

def ppf_test_infant_lr7(): # Node 07-1, load from 65000 step
    args = {}
    args = ppf_basic(args)
    args = infant_data(args)
    args = batch_size(args)

    args['col_name'] = 'infant_loss'
    args['exp_id'] = 'ppf_test_sample_lr7'
    args['load_exp'] = 'ppf/infant/train_lr7'
    args['pure_test'] = True
    args['test_meta_path'] = '/data4/shetw/infant_headcam/metafiles/infant_test_ppf_sample.meta'
    args['frame_root'] = '/data4/shetw/infant_headcam/jpgs_extracted'
    args['flip_frame'] = False # Not effective during training
    return args

def ppf_test_infant_lr7_a2(): # Node 07-7, load from 65000 step
    args = {}
    args = ppf_basic(args)
    args = infant_data(args)
    args = batch_size(args)

    args['col_name'] = 'infant_loss'
    args['exp_id'] = 'ppf_test_sample_lr7_a2'
    args['load_exp'] = 'ppf/infant/train_lr7'
    args['pure_test'] = True
    args['test_meta_path'] = '/data4/shetw/infant_headcam/metafiles/infant_test_ppf_sample_a2.meta'
    args['frame_root'] = '/data4/shetw/infant_headcam/jpgs_extracted'
    args['flip_frame'] = False # Not effective during training
    return args

def ppf_test_infant_lr7_a3(): # Node 07-1, load from 65000 step
    args = {}
    args = ppf_basic(args)
    args = infant_data(args)
    args = batch_size(args)

    args['col_name'] = 'infant_loss'
    args['exp_id'] = 'ppf_test_sample_lr7_a3'
    args['load_exp'] = 'ppf/infant/train_lr7'
    args['pure_test'] = True
    args['test_meta_path'] = '/data4/shetw/infant_headcam/metafiles/infant_test_ppf_sample_a3.meta'
    args['frame_root'] = '/data4/shetw/infant_headcam/jpgs_extracted'
    args['flip_frame'] = False # Not effective during training
    return args

def ppf_test_infant_lr7_s1(): # Node 07-5, load from 65000 step
    args = {}
    args = ppf_basic(args)
    args = infant_data(args)
    args = batch_size(args)

    args['col_name'] = 'infant_loss'
    args['exp_id'] = 'ppf_test_sample_lr7_s1'
    args['load_exp'] = 'ppf/infant/train_lr7'
    args['pure_test'] = True
    args['test_meta_path'] = '/data4/shetw/infant_headcam/metafiles/infant_test_ppf_sample_s1.meta'
    args['frame_root'] = '/data4/shetw/infant_headcam/jpgs_extracted'
    args['flip_frame'] = False # Not effective during training
    return args

def ppf_test_infant_lr7_s2(): # Node 07-6, load from 65000 step
    args = {}
    args = ppf_basic(args)
    args = infant_data(args)
    args = batch_size(args)

    args['col_name'] = 'infant_loss'
    args['exp_id'] = 'ppf_test_sample_lr7_s2'
    args['load_exp'] = 'ppf/infant/train_lr7'
    args['pure_test'] = True
    args['test_meta_path'] = '/data4/shetw/infant_headcam/metafiles/infant_test_ppf_sample_s2.meta'
    args['frame_root'] = '/data4/shetw/infant_headcam/jpgs_extracted'
    args['flip_frame'] = False # Not effective during training
    return args

def ppf_test_infant_lr7_s3(): # Node 07-5, load from 65000 step
    args = {}
    args = ppf_basic(args)
    args = infant_data(args)
    args = batch_size(args)

    args['col_name'] = 'infant_loss'
    args['exp_id'] = 'ppf_test_sample_lr7_s3'
    args['load_exp'] = 'ppf/infant/train_lr7'
    args['pure_test'] = True
    args['test_meta_path'] = '/data4/shetw/infant_headcam/metafiles/infant_test_ppf_sample_s3.meta'
    args['frame_root'] = '/data4/shetw/infant_headcam/jpgs_extracted'
    args['flip_frame'] = False # Not effective during training
    return args

def ppf_test_infant_lr58(): # Node 07-4, load from 22500 step
    args = {}
    args = ppf_basic(args)
    args = infant_data(args)
    args = batch_size(args)

    args['col_name'] = 'infant_loss'
    args['exp_id'] = 'ppf_test_sample_lr58'
    args['load_exp'] = 'ppf/infant/train_lr58'
    args['pure_test'] = True
    args['test_meta_path'] = '/data4/shetw/infant_headcam/metafiles/infant_test_ppf_sample.meta'
    args['frame_root'] = '/data4/shetw/infant_headcam/jpgs_extracted'
    args['flip_frame'] = False # Not effective during training
    return args