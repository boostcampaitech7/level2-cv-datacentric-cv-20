import os
import cv2
import shutil
from pathlib import Path
from argparse import ArgumentParser

LANGUAGE_LIST = ['japanese', 'chinese', 'thai', 'vietnamese']

def parse_args():

    script_dir = os.path.dirname(os.path.abspath(__file__))    
    default_data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=default_data_dir)
    parser.add_argument('--output_dir', type=str, default="gray_data")
    parser.add_argument('--val', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    return args

def convert_image_to_gray(image_path, save_path):
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Unable to process image: {image_path}")
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(save_path), gray)

    return True


def main(base_dir,output_dir,val):
    
    subsets =  ['train', 'test', 'val'] if val else ['train', 'test']
    base_dir = Path(base_dir)
    gray_base_dir = base_dir / output_dir

    for language in LANGUAGE_LIST:
        receipt_dir = base_dir / f"{language}_receipt"
        gray_receipt_dir = gray_base_dir / f"{language}_receipt" 
        
        for subset in subsets:
            (gray_receipt_dir / 'img' / subset).mkdir(parents=True, exist_ok=True)
        
        shutil.copytree(receipt_dir / 'ufo', gray_receipt_dir / 'ufo', dirs_exist_ok=True)
        
        for subset in subsets:
            img_dir = receipt_dir / 'img' / subset
            gray_img_dir = gray_receipt_dir / 'img' / subset
            
            for img_path in img_dir.glob('*'):
                gray_path = gray_img_dir / f"{img_path.name}"
                if convert_image_to_gray(img_path, gray_path):
                    print(f"Converting to grayscale and saving complete: {gray_path}")
    print("All processing has been completed.")

if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)