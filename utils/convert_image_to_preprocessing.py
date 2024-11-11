import os
import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

LANGUAGE_LIST = ['japanese', 'chinese', 'thai', 'vietnamese']

def parse_args():

    script_dir = os.path.dirname(os.path.abspath(__file__))    
    default_data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=default_data_dir)
    parser.add_argument('--output_dir', type=str, default="prep_data")
    parser.add_argument('--val', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    return args

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        return json.load(handle)

def convert_and_save_preprocessing(image_path, save_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Unable to process image: {image_path}")
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.medianBlur(gray, 5)#cv2.GaussianBlur(gray, (3, 3), 0)

    background = cv2.dilate(gray, np.ones((3,3), np.uint8), iterations=10)
    diff = cv2.absdiff(gray, background)
    diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    inverted_diff = cv2.bitwise_not(diff) 
    # binary = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 21, 3)
    cv2.imwrite(str(save_path), inverted_diff)
    return True

def main(base_dir, output_dir, val):
    
    base_dir = Path(base_dir)
    prep_dir = base_dir / output_dir
    subsets =  ['train', 'test', 'val'] if val else ['train', 'test']

    for language in LANGUAGE_LIST:
        receipt_dir = base_dir / f"{language}_receipt"
        gray_receipt_dir = prep_dir / f"{language}_receipt"  
        
        for subset in subsets:
            (gray_receipt_dir / 'img' / subset).mkdir(parents=True, exist_ok=True)
        
        shutil.copytree(receipt_dir / 'ufo', gray_receipt_dir / 'ufo', dirs_exist_ok=True)
        
        for subset in subsets:
            img_dir = receipt_dir / 'img' / subset
            gray_img_dir = gray_receipt_dir / 'img' / subset
            
            for img_path in img_dir.glob('*'):
                gray_path = gray_img_dir / f"{img_path.name}"
                if convert_and_save_preprocessing(img_path, gray_path):
                    print(f"Preprocessing conversion and saving completed: {gray_path}")

    print("All processing has been completed.")

if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)