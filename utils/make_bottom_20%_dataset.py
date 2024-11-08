import os
import cv2
import glob
import json
import shutil
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from shapely.affinity import rotate
from argparse import ArgumentParser
from shapely.geometry import Polygon
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

LANGUAGE_LIST = ['japanese', 'chinese', 'thai', 'vietnamese']
LANGUAGE_LIST_SUB = {'japanese':'ja','chinese':'zh', 'thai':'th', 'vietnamese':'vi'}
iou_data = {}
def parse_args():

    script_dir = os.path.dirname(os.path.abspath(__file__))    
    default_data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=default_data_dir)
    parser.add_argument('--output_dir', type=str, default="bottom_iou_data")
    parser.add_argument('--gt_path', type=str, default=default_data_dir)
    parser.add_argument('--pred_path', type=str, default=script_dir+"/pred_output.json")
    parser.add_argument('--is_compare_data', type=bool, default=False)
    parser.add_argument('--iou', type=int, default=30)
    parser.add_argument('--val', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    return args

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

def get_images_info(gt_path, pred_path):
    path_lists = glob.glob(f"{gt_path}/*_receipt/ufo/train.json")
    data = {}
    data['images'] = {}
    for path in path_lists:    
        json_data = read_json(path)
        images = list(json_data['images'].items())
        data['images'].update(dict(images))

    print(f"Total number of GT_images: {len(data['images'])}")
    print(gt_path)
    path_lists_100 = glob.glob(pred_path)
    data_train100 = {}
    data_train100['images'] = {}
    for path in path_lists_100:
        json_data = read_json(path)
        images = list(json_data['images'].items())
        data_train100['images'].update(dict(images))

    print(f"Total number of pred_images: {len(data_train100['images'])}")

    return data,data_train100

def get_keys(data):
    keys_list = sorted(list(data['images'].keys()))
    ja_keys_list = list()
    zh_keys_list = list()
    th_keys_list = list()
    vi_keys_list = list()

    for key in keys_list:
        if key.split('.')[1] == 'ja':
            ja_keys_list.append(key)
        elif key.split('.')[1] == 'zh':
            zh_keys_list.append(key)
        elif key.split('.')[1] == 'th':
            th_keys_list.append(key)
        elif key.split('.')[1] == 'vi':
            vi_keys_list.append(key)
    t_key_list = [sorted(ja_keys_list),sorted(zh_keys_list),sorted(th_keys_list),sorted(vi_keys_list)]
    for lan in t_key_list:
        print(len(lan))
    return t_key_list

def get_box_angle(box):
    coords = np.array(box.exterior.coords)
    edges = np.diff(coords, axis=0)
    longest_edge = edges[np.argmax(np.sum(edges**2, axis=1))]
    angle = np.arctan2(longest_edge[1], longest_edge[0])
    return angle

def angle_difference(box1, box2):
    angle1 = get_box_angle(box1)
    angle2 = get_box_angle(box2)
    diff = angle2 - angle1
    return (diff + np.pi) % (2 * np.pi) - np.pi

def rotated_boxes_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)    
    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)    
    inter_area = poly1.intersection(poly2).area    
    union_area = poly1.area + poly2.area - inter_area    
    iou = inter_area / union_area if union_area > 0 else 0    
    return iou

def store_data(language, image, box_num, box, box2, iou, angle_diff):
    if language not in iou_data:
        iou_data[language] = {}
    if image not in iou_data[language]:
        iou_data[language][image] = {}
    iou_data[language][image][box_num] = {'pbox':box, 'ybox':box2, 'iou': iou, 'angle_diff': angle_diff}

def get_data(language, image, box_num):
    return iou_data[language][image][box_num]

def save_iou_data(filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(iou_data, f, ensure_ascii=False, indent=4)

def load_iou_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        iou_data = json.load(f)
    return iou_data

def get_all_iou_values():
    iou_values = []

    for lang in iou_data:  
        for image in iou_data[lang]:  
            for box in iou_data[lang][image]:                   
                 iou_values.append(iou_data[lang][image][box]['iou'])  
    return iou_values


def get_percentile_iou(percentile):
    iou_values = get_all_iou_values()
    iou_values.sort()  

    index = int(len(iou_values) * percentile / 100)
    percentile_value = iou_values[index]  
    
    return percentile_value

def update_image_with_filtered_boxes(img_path, boxes_to_keep, save_path,blur_strength=51):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Unable to process image: {img_path}")
        return False  
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray_img.shape
    adjusted_blur_strength = np.max([3, int(blur_strength * (np.min([height, width]) / 1000))])
    if adjusted_blur_strength % 2 == 0:
        adjusted_blur_strength += 1

    blurred_img = cv2.GaussianBlur(gray_img, (adjusted_blur_strength, adjusted_blur_strength), 0)
    img_result = gray_img.copy() 
 
    for box in boxes_to_keep:
        pts = np.array(box, dtype=np.int32)
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], (255))
        img_result = cv2.bitwise_and(img_result, cv2.bitwise_not(mask))  
        blurred_no_box = cv2.bitwise_and(blurred_img, mask)       
        img_result = cv2.bitwise_or(img_result, blurred_no_box)

    cv2.imwrite(str(save_path), img_result)
    return True

def update_json_with_filtered_boxes(json_path, boxes_to_keep,receipt_dir,subset):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keys_to_delete = [] 
    for key, image_data in data.get('images', {}).items():
        new_words = {} 
        for word_id, word in image_data.get('words').items():            
            if word_id in boxes_to_keep:               
                new_words[word_id]=word
        if len(new_words) == 0 :
            image_file_path = os.path.join(receipt_dir, f'img/{subset}',f"{key}")
            if os.path.exists(image_file_path):
                os.remove(image_file_path)  
                print(new_words)
                print(f"Image file deleted: {image_file_path}")
                keys_to_delete.append(key) 
            else:
                print(f"Image file does not exist: {image_file_path}")                
        else:
            image_data['words'] = new_words 

    for key in keys_to_delete:
        if key in data['images']:
            del data['images'][key]
            print(f"Image {key} deleted")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def filter_and_remove_high_iou_boxes(language, data, bottom_iou, receipt_dir, bottom_iou_receipt_dir, subsets):
    for subset in subsets:
        img_dir = receipt_dir / 'img' / subset
        bottom_iou_img_dir = bottom_iou_receipt_dir / 'img' / subset
        boxes_to_keep_id_all = [] 
        for img_path in img_dir.glob('*'):        
            boxes_to_keep_id = []  
            boxes_to_keep_box = []             
            if subset != 'test':
                
                image = data[LANGUAGE_LIST_SUB[language]].get(str(img_path).split('/')[-1])
                for box in image:                  
                    d = image[box]
                    iou = d['iou']
                    if iou > bottom_iou: 
                        boxes_to_keep_box.append(d['pbox']) 
                    else:
                        boxes_to_keep_id.append(box)
            boxes_to_keep_id_all+=boxes_to_keep_id
            if subset != 'test':
                bottom_iou_path = bottom_iou_img_dir / f"{img_path.name}"
                #print(img_path.name)
                if update_image_with_filtered_boxes(img_path, boxes_to_keep_box, bottom_iou_path):
                    pass
                    #print(f"Filtered image saved at: {bottom_iou_path}")
  
            if subset == 'test':
                bottom_iou_path = bottom_iou_img_dir / f"{img_path.name}"
                shutil.copy(img_path, bottom_iou_path)  
                #print(f"Copied image at: {bottom_iou_path}")

        print("total len: ",len(boxes_to_keep_id_all))
        if subset != 'test':       
            json_path = bottom_iou_receipt_dir / 'ufo' / 'train.json'  
            update_json_with_filtered_boxes(json_path, boxes_to_keep_id_all,bottom_iou_receipt_dir,subset)
        
def get_bottom_n_per_of_iou(gt_path, pred_path, is_compare_data,iou):
    global iou_data
    gt_images, pred_images = get_images_info(gt_path, pred_path)
    t_key_list = get_keys(gt_images)
    iou_data_filename = 'iou_data.json'
    if not is_compare_data:
        for lang in t_key_list:
            country = lang[0].split('.')[1]
            for li in tqdm(lang):
                for p1 in gt_images['images'][li]['words']:
                    max = 0 
                    mp2 = None
                    box1 = gt_images['images'][li]['words'][p1]['points']
                    for p2 in pred_images['images'][li]['words']:
                        box2 = pred_images['images'][li]['words'][p2]['points']
                        iou = rotated_boxes_iou(box1, box2)
                        if max < iou:
                            max = iou
                            mp2 = p2
                    angle_diff = None
                    if not mp2 == None: 
                        box2 = pred_images['images'][li]['words'][mp2]['points']
                        angle_diff = angle_difference(Polygon(box1), Polygon(box2))
                    store_data(country,li,p1,box1,box2,max,angle_diff)
        save_iou_data(iou_data_filename)          
    else:
        iou_data = load_iou_data(iou_data_filename)
    
    top_iou = get_percentile_iou(100-iou)  
    print(f"Top  {100-iou}% IOU value: {top_iou}")

    bottom_iou = get_percentile_iou(iou)  
    print(f"Bottom {iou}% IOU value: {bottom_iou}")

    return bottom_iou

def main(base_dir, output_dir, val, gt_path, pred_path, is_compare_data,iou):
    base_dir = Path(base_dir)
    bottom_iou_base_dir = base_dir / output_dir 
    bottom_iou = get_bottom_n_per_of_iou(gt_path, pred_path, is_compare_data,iou)
    subsets =  ['train', 'test', 'val'] if val else ['train', 'test']
    for language in LANGUAGE_LIST:
        receipt_dir = base_dir / f"{language}_receipt"
        bottom_iou_receipt_dir = bottom_iou_base_dir / f"{language}_receipt"  

        for subset in subsets:
            (bottom_iou_receipt_dir / 'img' / subset).mkdir(parents=True, exist_ok=True)
        
        shutil.copytree(receipt_dir / 'ufo', bottom_iou_receipt_dir / 'ufo', dirs_exist_ok=True)
        
        filter_and_remove_high_iou_boxes(language, iou_data, bottom_iou, receipt_dir, bottom_iou_receipt_dir, subsets)
        print(f"{language} processing is complete.")
        
    print("All processing is complete.")
if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)