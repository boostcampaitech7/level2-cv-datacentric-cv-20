from glob import glob
import json

def coco_to_ufo(coco_json_path, ufo_json_path, lang):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    ufo_data = {"images": {}}

    # COCO 이미지 정보 처리
    for image in coco_data['images']:
        img_filename = image['file_name']
        ufo_data["images"][img_filename] = {
            "paragraphs": {},
            "words": {},
            "chars": {},
            "img_w": image['width'],
            "img_h": image['height'],
            "tags": [],
            "relations": {}
        }

    # COCO 어노테이션 정보 처리
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        img_filename = next(img['file_name'] for img in coco_data['images'] if img['id'] == image_id)
        annotation_id = str(annotation['id'])

        # 폴리곤 또는 bbox 처리
        points = []
        if "segmentation" in annotation and annotation["segmentation"]:
            # 폴리곤 형태로 변환
            segmentation = annotation['segmentation'][0]
            points = [[segmentation[i], segmentation[i + 1]] for i in range(0, len(segmentation), 2)]
        elif "bbox" in annotation:
            # Bounding box 형태 처리
            x, y, w, h = annotation['bbox']
            points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

        ufo_data["images"][img_filename]["words"][annotation_id] = {
            "transcription": annotation.get('caption', ''),
            "points": points
        }

    with open(ufo_json_path[0], 'w', encoding='utf-8') as f:
        json.dump(ufo_data, f, ensure_ascii=False, indent=4)

    print(f"{lang} UFO 포맷이 COCO 포맷으로 변환되었습니다.")
# 사용 예시
# 변환된 UFO JSON 파일 경로


LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']
for lang in LANGUAGE_LIST:
    # UFO JSON 파일 경로
    ufo_file_path = glob(f'./data/{lang}_receipt/ufo/train.json')
# 변환된 COCO JSON 파일 경로
    coco_file_path = f'./data/{lang}_receipt/ufo/coco_cvat.json'

    coco_to_ufo(coco_file_path, ufo_file_path, lang)