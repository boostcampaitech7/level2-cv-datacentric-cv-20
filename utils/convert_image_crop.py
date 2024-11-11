import os
import cv2
import shutil
import json
from pathlib import Path
from argparse import ArgumentParser

LANGUAGE_LIST = ['japanese', 'chinese', 'thai', 'vietnamese']

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    default_data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=default_data_dir)
    parser.add_argument('--output_dir', type=str, default="cropped_data")
    parser.add_argument('--padding', type=int, default=50)
    parser.add_argument('--val', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    return args

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        return json.load(handle)

def write_json(filename, data):
    with Path(filename).open('w', encoding='utf8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

def get_bounding_box(points, img_width, img_height, padding=50):
    if not points:
        return [0, 0, img_width, img_height]
    
    x_coordinates, y_coordinates = zip(*points)
    x1 = max(0, min(x_coordinates) - padding)
    y1 = max(0, min(y_coordinates) - padding)
    x2 = min(img_width, max(x_coordinates) + padding)
    y2 = min(img_height, max(y_coordinates) + padding)
    
    return [x1, y1, x2, y2]

def crop_image(image_path, bounding_box):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Unable to read image: {image_path}")
        return None
    x1, y1, x2, y2 = map(int, bounding_box)
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img

def process_images_and_json(json_data, img_dir, save_img_dir, padding=0):
    updated_json_data = {'images': {}}
    
    for img_name, img_data in json_data['images'].items():
        all_points = []
        for bbox in img_data['words'].values():
            all_points.extend(bbox['points'])
        
        img_width, img_height = img_data['img_w'], img_data['img_h']
        x1, y1, x2, y2 = get_bounding_box(all_points, img_width, img_height, padding)
        
        image_path = img_dir / img_name
        try:
            cropped_img = crop_image(image_path, [x1, y1, x2, y2])
            if cropped_img is None:
                print(f"Unable to process image: {image_path}")
                continue
            
            save_image_path = save_img_dir / img_name
            cv2.imwrite(str(save_image_path), cropped_img)
            
            updated_img_data = img_data.copy()
            updated_img_data['original_img_w'] = img_width
            updated_img_data['original_img_h'] = img_height
            updated_img_data['img_w'] = x2 - x1
            updated_img_data['img_h'] = y2 - y1
            updated_img_data['crop_x1'] = x1
            updated_img_data['crop_y1'] = y1
            updated_img_data['crop_x2'] = x2
            updated_img_data['crop_y2'] = y2
            updated_img_data['words'] = {}
            for word_id, bbox in img_data['words'].items():
                updated_bbox = bbox.copy()
                updated_bbox['points'] = [[x-x1, y-y1] for x, y in bbox['points']]
                updated_img_data['words'][word_id] = updated_bbox
            
            updated_json_data['images'][img_name] = updated_img_data
            
            print(f"Image crop completed: {save_image_path}")
        except Exception as e:
            print(f"Error occurred while processing image: {image_path}, Error: {str(e)}")
            continue
    
    return updated_json_data

def main(base_dir, output_dir, padding, val):
    base_dir = Path(base_dir)
    cropped_base_dir = base_dir / output_dir
    padding = padding

    for language in LANGUAGE_LIST:
        receipt_dir = base_dir / f"{language}_receipt"
        cropped_receipt_dir = cropped_base_dir / f"{language}_receipt"
        
        (cropped_receipt_dir / 'img' / 'train').mkdir(parents=True, exist_ok=True)
        
        (cropped_receipt_dir / 'ufo').mkdir(parents=True, exist_ok=True)
        
        json_path = receipt_dir / 'ufo' / "train.json"
        img_dir = receipt_dir / 'img' / 'train'
        cropped_img_dir = cropped_receipt_dir / 'img' / 'train'
        
        json_data = read_json(json_path)
        
        updated_json_data = process_images_and_json(json_data, img_dir, cropped_img_dir, padding)
        
        cropped_json_path = cropped_receipt_dir / 'ufo' / "train.json"
        write_json(cropped_json_path, updated_json_data)
        
        print(f"{language}  train processing completed")
        
        shutil.copytree(receipt_dir / 'img' / 'test', cropped_receipt_dir / 'img' / 'test', dirs_exist_ok=True)
        shutil.copy2(receipt_dir / 'ufo' / "test.json", cropped_receipt_dir / 'ufo' / "test.json")
        if val:
            shutil.copytree(receipt_dir / 'img' / 'val', cropped_receipt_dir / 'img' / 'val', dirs_exist_ok=True)
            shutil.copy2(receipt_dir / 'ufo' / "val.json", cropped_receipt_dir / 'ufo' / "val.json")
        
        print(f"{language} test data copy completed")

    print("All processing has been completed.")

if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)