"""
Generate the metafile of the infant_headcam data.
Note: Only the videos in enough length are chosen 
"""
from __future__ import print_function
import os
import argparse
from tqdm import tqdm

MIN_NUM_FRAMES = 4500 # = 3 mins

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
            '--save_file',
            default='/data/shetw/infant_headcam/metafiles/infant_3min_ppf.meta',
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
                record = "{} {}\n".format(os.path.join(infant_name, video_name), 
                                        len(os.listdir(frames_dir)))
                records.append(record)
    records.sort()
    return records


def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.save_file, 'w+') as mt_file:
        records = gen_metafile_record(args)
        mt_file.writelines(records)
    mt_file.close()

    
if __name__ == "__main__":
    main()

