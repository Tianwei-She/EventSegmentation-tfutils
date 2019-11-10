"""
Generate the metafile of the infant_headcam data.
Note: Only the videos in enough length are chosen 
"""
from __future__ import print_function
import os
import argparse
from random import sample
from tqdm import tqdm

MIN_NUM_FRAMES = 4500 # = 3 mins
TRAIN_TOTAL_NUM_FRAMES = 6840000 # = 76 hrs

def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to generate the metafile \
                of the infant_headcam data.')
    parser.add_argument(
            '--image_root_dir',
            default='/data/shetw/infant_headcam/jpgs_extracted',
            type=str, action='store',
            help='Directory to hold all the frames')
    parser.add_argument(
            '--save_file_train',
            default='/data/shetw/infant_headcam/metafiles/infant_train_ppf.meta',
            type=str, action='store',
            help='The saving path of the metafile')
    parser.add_argument(
            '--save_file_test',
            default='/data/shetw/infant_headcam/metafiles/infant_test_ppf.meta',
            type=str, action='store',
            help='The saving path of the metafile')
    return parser

 
def gen_metafile_record(args):

    records = []
    image_root_dir = args.image_root_dir
    for infant_name in os.listdir(image_root_dir):
        infant_dir = os.path.join(image_root_dir, infant_name)
        for video_name in tqdm(os.listdir(infant_dir)):
            frames_dir = os.path.join(infant_dir, video_name)
            # Check if there are enough number of frames
            if len(os.listdir(frames_dir)) > MIN_NUM_FRAMES:
                records.append((os.path.join(infant_name, video_name), len(os.listdir(frames_dir))))
                """
                record = "{} {}\n".format(os.path.join(infant_name, video_name), 
                                        len(os.listdir(frames_dir)))
                records.append(record)
                """
    # records.sort()
    return records

def split_train_test(records):
    shuffled_records = sample(records, len(records))
    total_length = 0
    for i, record in enumerate(records):
        vd_name, vd_len = record
        total_length += vd_len
        if total_length > TRAIN_TOTAL_NUM_FRAMES:
            break
    train_records = ["{} {}\n".format(vd_name, vd_len) for vd_name, vd_len in shuffled_records[:i]]
    test_records = ["{} {}\n".format(vd_name, vd_len) for vd_name, vd_len in shuffled_records[i:]]
    return train_records, test_records

def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.save_file_train, 'w+') as train_file:
        with open(args.save_file_test, 'w+') as test_file:
            records = gen_metafile_record(args)
            train_records, test_records = split_train_test(records)
            train_file.writelines(train_records)
            test_file.writelines(test_records)

    
if __name__ == "__main__":
    main()

