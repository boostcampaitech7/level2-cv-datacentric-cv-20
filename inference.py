import os
import os.path as osp
import json
import sys
import yaml
from glob import glob

import torch
import cv2
from torch import cuda
from tqdm import tqdm

from base.detect import detect
from base.model import EAST

from utils.argeParser import parse_args

CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

def do_inference(input_size, batch_size, data_dir, model_dir, pth_path, device, output_dir, output_fname, split='test'):
    model = EAST(pretrained=False).to(device)

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(model_dir, pth_path+'.pth')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []
    images = []

    print('Inference in progress')
    
    for image_fpath in tqdm(sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], [])):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    with open(osp.join(output_dir, output_fname+'.csv'), 'w') as f:
        json.dump(ufo_result, f, indent=4)


def main(args):
    do_inference(**args.__dict__)

if __name__ == '__main__':
    args = parse_args('inference')
    main(args)
