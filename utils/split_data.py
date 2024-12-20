from pathlib import Path
import json
import glob
import random
import os
import shutil

SEED = 20 
SPLIT_RATIO = 0.8 

def read_json(path: str):
    with Path(path).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

def make_json(data, path: str):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def split_data(path, lang):
    json_path = os.path.join(path, 'ufo', f'{lang}_train_relable_ufo.json')
    json_data = read_json(json_path)
    image_keys = list(json_data['images'].keys())

    random.seed(SEED)
    random.shuffle(image_keys)

    split_index= int(len(image_keys) * SPLIT_RATIO)
    train_keys = image_keys[:split_index]
    valid_keys = image_keys[split_index:]
    
    train_data = {
        'images' : {
            k : json_data['images'][k] for k in train_keys
        }
    }
    valid_data = {
        'images' : {
            k : json_data['images'][k] for k in valid_keys
        }
    }

    train_json_path = os.path.join(path, 'ufo', 'train_relabel.json')
    make_json(train_data, train_json_path)

    valid_json_path = os.path.join(path, 'ufo','val_relabel.json') 
    make_json(valid_data, valid_json_path)

def main():
    lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
    lang_name = ['chinese', 'japanese', 'thail', 'vietnamese']
    path_lists = [f'./data/{lang}_receipt' for lang in lang_list]
    for path, lang in zip(path_lists, lang_name):
        split_data(path, lang)

if __name__ == '__main__':
    main()