import numpy as np
import json
from glob import glob

def ufo_to_coco(ufo_file_path, coco_file_path, lang):
# UFO 형식의 JSON 로드
    with open(ufo_file_path[0], 'r', encoding='utf-8') as f:
        ufo_data = json.load(f)

    # COCO 형식 데이터 구조 초기화
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 카테고리 추가 (여기서는 기본적인 예시로 단일 카테고리를 추가)
    category_id = 1
    coco_data["categories"].append({
        "id": category_id,
        "name": "text",
        "supercategory": "text"
    })

    # 이미지와 어노테이션 변환
    image_id = 1
    annotation_id = 1

    for image_name, image_info in ufo_data["images"].items():
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": image_info["img_w"],
            "height": image_info["img_h"]
        })
        
        # 각 단어에 대해 어노테이션 추가
        for word_id, word_info in image_info["words"].items():
            points = word_info["points"]
            # Bounding box 계산: (x_min, y_min, width, height)
            x_min = min(point[0] for point in points)
            y_min = min(point[1] for point in points)
            width = max(point[0] for point in points) - x_min
            height = max(point[1] for point in points) - y_min
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, width, height],
                "segmentation": np.asarray(points).reshape(1,-1).tolist(),
                "area": width * height,
                "iscrowd": 0
            })
            annotation_id += 1
        
        image_id += 1

    # 변환된 COCO JSON 파일 저장
    with open(coco_file_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=4)

    print(f"{lang} UFO 포맷이 COCO 포맷으로 변환되었습니다.")

LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']
for lang in LANGUAGE_LIST:
# UFO JSON 파일 경로
    ufo_file_path = glob(f'./data/{lang}_receipt/ufo/train.json')
# 변환된 COCO JSON 파일 경로
    coco_file_path = f'./data/{lang}_receipt/ufo/coco_cvat.json'
    ufo_to_coco(ufo_file_path, coco_file_path, lang)
