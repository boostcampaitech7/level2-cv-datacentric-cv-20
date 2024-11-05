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

def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    
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

    return ufo_result


def main(device, model_dir, output_dir, data_dir, input_size, batch_size):
    # Initialize model
    model = EAST(pretrained=False).to(device)

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(model_dir, 'epoch_47_loss_1.3602.pth')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, data_dir,input_size,
                                batch_size, split='test')
    ufo_result['images'].update(split_result['images'])

    output_fname = 'baseline_8_50.csv'
    with open(osp.join(output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args('inference')
    main(**args.__dict__)
