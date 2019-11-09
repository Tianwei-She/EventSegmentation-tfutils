"""
Generate the frame-level metafile of the breakfast dataset.
"""

import os
import argparse
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to generate the metafile \
                of the breakfast data.')
    parser.add_argument(
            '--image_root_dir',
            default='/data4/shetw/breakfast/test_extracted_frames',
            type=str, action='store',
            help='Directory to hold all the frames')
    parser.add_argument(
            '--save_file',
            default='/data4/shetw/breakfast/metafiles/test_videos_metafile.txt',
            type=str, action='store',
            help='The saving path of the metafile')
    return parser

 
def gen_metafile_record(args):
    records = []
    image_root_dir = args.image_root_dir
    for video_name in tqdm(os.listdir(image_root_dir)):
        frames_dir = os.path.join(image_root_dir, video_name)
        record = "{} {}\n".format(video_name, len(os.listdir(frames_dir)))
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

