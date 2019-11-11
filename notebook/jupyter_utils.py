import pymongo as pm
import gridfs
import cPickle

import numpy as np
import matplotlib.pyplot as plt
import pylab

from scipy import misc
import os
import time

import sklearn.linear_model
import math
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
from scipy.stats import spearmanr

vis_big_dict = {}


def show_val(
        curr_expid, 
        conn, 
        key='loss', 
        dbname='combinet-test', 
        valid_key='topn',
        colname='combinet.files', 
        gridfs_name='combinet', 
        big_dict=vis_big_dict, 
        batch_watch_start=0,
        batch_watch_end=None,
        new_figure=True, 
        val_N=0, 
        batch_size=8, 
        label_now=None, 
        val_step=1.0, 
        batch_offset=0,
        do_conv=None,
        do_plot=True,
        special_delete=None):
    
    if label_now is None:
        label_now = curr_expid

    find_res = conn[dbname][colname].find({'exp_id': curr_expid,
                                            'validation_results': {'$exists': True}})
    find_res = sorted(find_res, key = lambda x: x['step'])
    if len(find_res)==0:
        return None
    new_find_res = []
    for curr_indx in xrange(len(find_res)-1):
        if find_res[curr_indx]['step'] == find_res[curr_indx+1]['step'] and find_res[curr_indx]['step'] is not None:
            continue
        new_find_res.append(find_res[curr_indx])
    new_find_res.append(find_res[len(find_res)-1])
    find_res = new_find_res

    if len(find_res)==0:
        return
    find_res = filter(lambda x: valid_key in x['validation_results'], find_res)
    if special_delete:
        del find_res[special_delete]
    if key in find_res[0]['validation_results'][valid_key].keys():
        list_res = find_res
    else:
        print(find_res[0]['validation_results'][valid_key].keys())
        assert key in find_res[0]['validation_results'][valid_key].keys(), 'Wrong key %s!' % key
    test_vec = [r['validation_results'][valid_key][key] for r in list_res]
    x_range = range(len(test_vec))
    _N = val_N
    #print(test_vec)
    if new_figure and do_plot:
        plt.figure(figsize=(9, 5))
    x_range = (np.asarray(x_range) + 1.0)*val_step*(batch_size)/(8) + batch_offset
    x_range = np.asarray(x_range[_N:])
    test_vec = np.asarray(test_vec[_N:])

    if do_conv:
        conv_list = np.ones([do_conv]) / do_conv
        test_vec = np.convolve(test_vec, conv_list, mode='valid')
        x_range = x_range[:len(test_vec)]

    choose_indx = x_range > batch_watch_start
    if batch_watch_end is not None:
        choose_indx = choose_indx & (x_range < batch_watch_end)

    if do_plot:
        plt.plot(x_range[choose_indx], test_vec[choose_indx], label = label_now)
    if 'top' in key and  do_plot:
        print(curr_expid, key, np.max(test_vec[choose_indx]), np.argmax(test_vec))
    if do_plot:
        plt.title('Validation Performance, %s' % key)    

    plt.legend(loc = 'best')

    if do_plot:
        return test_vec[_N:]
    else:
        return x_range[choose_indx], test_vec[choose_indx]


def show_train_learnrate(
        curr_expid, 
        conn, 
        cache_dict={},
        dbname='combinet-test', 
        colname='combinet.files', 
        start_N=50, 
        with_dataset=None, 
        batch_watch_start=0,
        batch_watch_end=None,
        do_conv=False, 
        conv_len=100, 
        new_figure=True, 
        batch_size=8, 
        batch_offset=0, 
        max_step=None, 
        label_now=None,
        loss_key='loss',
        refresh_cache=True,
        ):
    
    if label_now is None:
        label_now = curr_expid

    cache_key = os.path.join(dbname, colname, curr_expid)
    if refresh_cache or cache_key not in cache_dict:
        find_res = conn[dbname][colname].find(
                    {'exp_id': curr_expid, 'train_results': {'$exists': True}})
        find_res = sorted(find_res, key = lambda x: x['step'])
        new_find_res = []
        for curr_indx in xrange(len(find_res)-1):
            if find_res[curr_indx]['step'] == find_res[curr_indx+1]['step']:
                continue
            new_find_res.append(find_res[curr_indx])
        new_find_res.append(find_res[len(find_res)-1])
        find_res = new_find_res
        cache_dict[cache_key] = find_res
    else:
        find_res = cache_dict[cache_key]

    if max_step:
        find_res = filter(lambda x: x['step']<max_step, find_res)
    if with_dataset is None:
        train_vec = np.concatenate([[(_r[loss_key], _r['learning_rate']) for _r in r['train_results']] 
                    for r in find_res])
    else:
        train_vec = np.concatenate([[_r[loss_key] for _r in r['train_results']] 
                    for r in find_res])
        rate_vec = np.concatenate([[_r['learning_rate'] for _r in r['train_results']] 
                    for r in find_res])
        all_dataset_label = np.concatenate([[_r['dataset'] for _r in r['train_results']] 
                    for r in find_res])
        
        all_rec_dict = {}

        for curr_dataset in with_dataset:
            all_rec_dict[curr_dataset] = []
        for rec_indx in xrange(train_vec.shape[0]):
            curr_dataset = all_dataset_label[rec_indx]
            if curr_dataset in with_dataset:
                all_rec_dict[curr_dataset].append(train_vec[rec_indx])
        min_len = train_vec.shape[0]
        for curr_dataset in np.unique(with_dataset):
            curr_count = with_dataset.count(curr_dataset)
            curr_len = len(all_rec_dict[curr_dataset])//curr_count * curr_count
            all_rec_dict[curr_dataset] = np.mean(np.asarray(all_rec_dict[curr_dataset][:curr_len]).reshape([-1, curr_count]), axis = 1)

            if len(all_rec_dict[curr_dataset]) < min_len:
                min_len = len(all_rec_dict[curr_dataset])

        new_train_vec = np.zeros([min_len, 2])

        for curr_dataset in np.unique(with_dataset):
            new_train_vec[:, 0] = new_train_vec[:, 0] + all_rec_dict[curr_dataset][:min_len]

        new_train_vec[:, 1] = rate_vec[::len(with_dataset)][:min_len]

        train_vec = new_train_vec
    
    #print(train_vec.shape)
    _N = start_N
    if new_figure:
        fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    inter_list = train_vec[_N:, 0]
    inter_list = np.asarray(inter_list)
    inter_list = inter_list[inter_list < 50]
    if do_conv:
        conv_list = np.ones([conv_len])/conv_len
        inter_list = np.convolve(inter_list, conv_list, mode='valid')
    
    temp_x_list = np.asarray(range(len(inter_list)))*1.0*batch_size/(10000*8) + batch_offset
    new_indx_list = temp_x_list > batch_watch_start
    if batch_watch_end is not None:
        new_indx_list = (temp_x_list>batch_watch_start) & (temp_x_list<batch_watch_end)
    plt.plot(temp_x_list[new_indx_list], inter_list[new_indx_list], label = label_now)
    plt.title('Training loss')
    plt.legend(loc = 'best')
    plt.subplot(1, 2, 2)
    temp_y_list = train_vec[_N:, 1]
    temp_x_list_2 = np.asarray(range(len(temp_y_list)))*1.0*batch_size/(10000*8) + batch_offset
    new_indx_list_2 = temp_x_list_2 > batch_watch_start
    if batch_watch_end is not None:
        new_indx_list_2 = (temp_x_list_2>batch_watch_start) & (temp_x_list_2<batch_watch_end)
    plt.plot(temp_x_list_2[new_indx_list_2], temp_y_list[new_indx_list_2], label = label_now)
    plt.title('Learning Rate')


def get_cached_or_load(idval, conn, dbname = 'combinet-test', 
                       colname = 'combinet.files', gridfs_name = 'combinet', big_dict = vis_big_dict):
    coll = conn[dbname][colname]
    fn = coll.find({'item_for': idval})[0]['filename']

    if fn in big_dict:
        saved_data = big_dict[fn]
    else:
        fs = gridfs.GridFS(coll.database, gridfs_name)
        fh = fs.get_last_version(fn)
        saved_data = cPickle.loads(fh.read())
        fh.close()
        
        big_dict[fn] = saved_data
        
    return saved_data
