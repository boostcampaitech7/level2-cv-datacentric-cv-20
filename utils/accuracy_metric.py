import os.path as osp
import json
from glob import glob

import cv2
from tqdm import tqdm

from base.detect import detect

def get_pred_bboxes(model, data_dir, input_size, batch_size, split='val'):
    '''
    model 수행 후 prediction bboxes 좌표
    ex) 'image1' : [[좌표1, 좌표2, 좌표3, 좌표4], [], ..]    
    '''
    lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']

    image_fnames, by_sample_bboxes = [], []
    images = []
    
    for image_fpath in tqdm(sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in lang_list], [])):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    pred_result = dict()
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        pred_result[image_fname] = bboxes

    return pred_result

def get_gt_bboxes(data_dir, split='val'):
    '''
    gt bboxes 좌표
    ex) 'image1' : [[좌표1, 좌표2, 좌표3, 좌표4], [], ..]
    '''
    lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']

    gt_result = dict()

    for nation in lang_list:
        with open(osp.join(data_dir, f'{nation}_receipt/ufo/{split}_relabel.json'), 'r', encoding='utf-8') as f:
            anno = json.load(f)
        for image in anno['images']:
            gt_result[image] = []
            for id in anno['images'][image]['words']:
                points = anno['images'][image]['words'][id]['points']
                gt_result[image].append(points)
    
    return gt_result

def get_lang_pred_bboxes(model, lang, data_dir, input_size, batch_size, split='val'):
    '''
    model 수행 후 lang 별 prediction bboxes 좌표
    ex) 'image1' : [[좌표1, 좌표2, 좌표3, 좌표4], [], ..]    

    lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
    '''
    val_json_path = osp.join(data_dir, f'{lang}_receipt/ufo/val_relabel.json')
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)

    val_image_fnames = val_data['images'].keys()
    
    image_paths = [osp.join(data_dir, f'{lang}_receipt/img/train', fname) for fname in val_image_fnames]
    
    image_fnames, by_sample_bboxes = [], []
    images = []

    for image_fpath in tqdm(image_paths, desc=f'Processing {lang} images'):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    pred_result = dict()
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        pred_result[image_fname] = bboxes
    return pred_result

def get_lang_gt_bboxes(data_dir, lang, split='val'):
    '''
    lang 별 gt bboxes 좌표
    ex) 'image1' : [[좌표1, 좌표2, 좌표3, 좌표4], [], ..]
    '''
    gt_result = dict()

    with open(osp.join(data_dir, f'{lang}_receipt/ufo/{split}_relabel.json'), 'r', encoding='utf-8') as f:
        anno = json.load(f)
    for image in anno['images']:
        gt_result[image] = []
        for id in anno['images'][image]['words']:
            points = anno['images'][image]['words'][id]['points']
            gt_result[image].append(points)
    return gt_result