import os
import time
import random

import numpy as np
from PIL import Image
from random import sample

IMAGENET_MEAN = np.array([123.68, 116.779, 103.939])

class FrameDataset():
    def __init__(
            self, frame_root, meta_path, 
            batch_size, num_frames, flip_frame=False, file_tmpl="Frame_{:06d}.jpg", crop_size=224, shuffle=False):
        
        self.frame_root = frame_root
        self.meta_path = meta_path
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.flip_frame = flip_frame # Flip top-bottom for infant videos
        self.file_tmpl = file_tmpl
        self.shuffle = shuffle
        self.num_batch_per_epoch = None
        self.video_list = self._parse_list() # A list of (index, vd_name, vd_length)
    
    def _parse_list(self):
        with open(self.meta_path, 'r') as f:
            lines = f.readlines()
            video_list = [(i, line.split()[0], int(line.split()[1])) 
                                for i, line in enumerate(lines)]
            self.num_batch_per_epoch = len(video_list)//self.batch_size
            print("Number of videos: {}".format(len(video_list)))
        return video_list
    
    def _get_frames_at_step(self, batch, step):
        """ Returns the step-th resized & normalized frame of each video in the batch
        Args:
            batch: a list of video information
            step: starts from 0
        Returns:
            An array in size [batch_size, height, width, 3]
        """
        frame_batch_list, index_list = [], []
        for video in batch:
            # pdb.set_trace()
            vd_index, vd_name, vd_length = video
            real_step = step % vd_length
            frame_path = os.path.join(self.frame_root, vd_name, 
                                        self.file_tmpl.format(real_step+1))
            
            # Preprocessing
            image = Image.open(frame_path)
            image = image.resize((self.crop_size, self.crop_size), Image.ANTIALIAS)
            if self.flip_frame:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image = np.array(image)
            image = np.subtract(image, IMAGENET_MEAN)
            
            frame_batch_list.append(image)
            index_list.append(vd_index)
        
        frame_batch = np.stack(frame_batch_list)
        index_batch = np.array(index_list)
        return (frame_batch, index_batch, step)

    def batch_of_frames_generator(self):
        """Yeild a batch of frames with indices at each time step."""
        if self.shuffle:
            video_list = sample(self.video_list, len(self.video_list))
        else:
            video_list = self.video_list
        
        for batch_i in range(self.num_batch_per_epoch):
            batch = video_list[batch_i*self.batch_size:(batch_i+1)*self.batch_size]
            for frame_i in range(self.num_frames+1): # 1 more initialization step
                yield self._get_frames_at_step(batch, frame_i)