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

def split_data(path):
    json_path = os.path.join(path, 'ufo', 'train.json')
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

    train_json_path = json_path
    make_json(train_data, train_json_path)

    valid_json_path = train_json_path.replace('train.json', 'val.json')  
    make_json(valid_data, valid_json_path)

    folder_path = os.path.join(path, 'img')
    train_folder_path = os.path.join(folder_path, 'train')
    valid_folder_path = os.path.join(folder_path, 'val')
    Path(valid_folder_path).mkdir(exist_ok=True)
    
    for k in valid_keys:
        shutil.move(os.path.join(train_folder_path, k), os.path.join(valid_folder_path, k))

def main():
    path_lists = glob.glob(f"./data/*_receipt")
    for path in path_lists:
        split_data(path)

if __name__ == '__main__':
    main()