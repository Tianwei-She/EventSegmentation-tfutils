"""
Generate the seperate metafiles for each baby.
Note: Only the videos in enough length are chosen 
"""
from __future__ import print_function
import os
import argparse
from random import sample
from tqdm import tqdm
import numpy as np

MIN_NUM_FRAMES = 7500 # = 5min

def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to generate the metafile \
                of the infant_headcam data.')
    parser.add_argument(
            '--image_root_dir',
            default='/data4/shetw/infant_headcam/jpgs_extracted',
            type=str, action='store',
            help='Directory to hold all the frames')
    parser.add_argument(
            '--baby_name',
            default='Samcam',
            type=str, action='store',
            help='Folder name of the baby')        
    parser.add_argument(
            '--save_file',
            default='/data4/shetw/infant_headcam/metafiles/infant_sam.meta',
            type=str, action='store',
            help='The saving path of the metafile')
    return parser
 
def gen_metafile_record(args):
    records = []
    all_num_frames = []
    image_root_dir = args.image_root_dir
    for infant_name in os.listdir(image_root_dir):
        if infant_name != args.baby_name:
            continue
        infant_dir = os.path.join(image_root_dir, infant_name)
        for video_name in tqdm(os.listdir(infant_dir)):
            frames_dir = os.path.join(infant_dir, video_name)
            # Check if there are enough number of frames
            num_frames = len(os.listdir(frames_dir))
            if num_frames > MIN_NUM_FRAMES:
                record = "{} {}\n".format(os.path.join(infant_name, video_name), 
                                        num_frames)
                records.append(record)
                all_num_frames.append(num_frames)
    records.sort()
    print("Average number of frames: {}".format(np.mean(all_num_frames)))
    return records

def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.save_file, 'w+') as train_file:
        records = gen_metafile_record(args)
        train_file.writelines(records)

    
if __name__ == "__main__":
    main()

