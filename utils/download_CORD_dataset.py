import os
import json
import shutil

from datasets import load_dataset

def cord_json_to_ufo(type, dataset, output_path):
    train_json_data = dataset['train']['ground_truth']
    val_json_data = dataset['validation']['ground_truth']
    test_json_data = dataset['test']['ground_truth']

    ufo_format = {'images' : {}}
    if type == 'train':
        json_data = train_json_data
    elif type == 'val':
        json_data = val_json_data
    else:
        json_data = test_json_data
        
    for id, img_data in enumerate(json_data):
        img_name = f'CORD_dataset_{type}_{str(id).zfill(4)}.png'

        if img_name not in ufo_format['images']:
            ufo_format['images'][img_name] = {'words': {}}
        
        word_id = 0

        for word_data in eval(img_data)['valid_line']:
            for word_info in word_data['words']:

                quad = word_info['quad']
                points = [
                    [quad['x1'], quad['y1']],
                    [quad['x2'], quad['y2']],
                    [quad['x3'], quad['y3']],
                    [quad['x4'], quad['y4']]
                ]

                word_index = f'{str(word_id+1).zfill(4)}'

                ufo_annotation = {
                    'transcription': word_info['text'],
                    'points': points
                }

                ufo_format['images'][img_name]['words'][word_index] = ufo_annotation

                word_id += 1

        ufo_format['images'][img_name]['img_w'] = eval(img_data)['meta']['image_size']['width']
        ufo_format['images'][img_name]['img_h'] = eval(img_data)['meta']['image_size']['height']

    with open(output_path, 'w') as f:
        json.dump(ufo_format, f, indent=4)

def merge_train_val_json(train_path, val_path):
    with open(train_path, "r") as f:
        train_data = json.load(f)


    with open(val_path, "r") as f:
        val_data = json.load(f)
        for key, image_data in val_data["images"].items():
            train_data["images"][key] = image_data

        os.remove(val_path)
        print("val.json 삭제완료")

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=4)

    print("train.json 완성")



def main():
    print('데이터셋 불러오는 중')
    dataset = load_dataset("naver-clova-ix/cord-v2")

    train_img_data = dataset['train']['image']
    val_img_data = dataset['validation']['image']
    test_img_data = dataset['test']['image']

    img_path = './data/cord_receipt/img'
    json_path = './data/cord_receipt/ufo'

    # img 다운로드 과정
    train_path = os.path.join(img_path, 'train')
    test_path = os.path.join(img_path, 'test')

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    print('train data 다운로드 중')
    for i in range(len(train_img_data)):
        file_name = f'{train_path}/CORD_dataset_train_{str(i).zfill(4)}.png'
        train_img_data[i].save(file_name)

    print('val data 다운로드 중')
    for i in range(len(val_img_data)):
        file_name = f'{train_path}/CORD_dataset_val_{str(i).zfill(4)}.png'
        val_img_data[i].save(file_name)

    print('test data 다운로드 중')
    for i in range(len(test_img_data)):
        file_name = f'{test_path}/CORD_dataset_test_{str(i).zfill(4)}.png'
        test_img_data[i].save(file_name)

    # json 다운로드 과정
    os.makedirs(json_path, exist_ok=True)

    train_json_path = json_path+'/train.json'
    val_json_path = json_path+'/val.json'
    test_json_path = json_path+'/test.json'

    print('json 파일 다운로드 중')
    cord_json_to_ufo('train', dataset, train_json_path)
    cord_json_to_ufo('val', dataset, val_json_path)
    cord_json_to_ufo('test', dataset, test_json_path)

    merge_train_val_json(train_json_path, val_json_path)

if __name__ == '__main__':
    main()