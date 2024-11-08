import os
import json
import shutil
from typing import List

def load_text_files(text_files_dir: str) -> List[str]:
    """텍스트 파일 경로를 리스트로 반환합니다."""
    return [f for f in os.listdir(text_files_dir) if f.endswith('.txt')]

def parse_text_file(file_path: str) -> List[dict]:
    """텍스트 파일을 읽어 UFO 형식으로 변환합니다."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    words = {}
    for idx, line in enumerate(lines):
        coords = list(map(float, line.split(',')[:8]))  # 8개 좌표 추출
        transcription = line.split(',')[-1].strip()     # 텍스트 추출
        words[str(idx).zfill(4)] = {
            "transcription": transcription,
            "points": [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
        }
    return words

def create_ufo_data(text_files_dir: str, img_width: int, img_height: int) -> dict:
    """UFO 형식 데이터를 생성하여 반환합니다."""
    merged_ufo_data = {"images": {}}
    text_files = load_text_files(text_files_dir)

    for text_file in text_files:
        image_name = text_file.replace('.txt', '.jpg')
        text_file_path = os.path.join(text_files_dir, text_file)

        # 각 이미지의 UFO 데이터 초기화
        ufo_image_data = {
            "paragraphs": {}, "words": {}, "chars": {}, "img_w": img_width,
            "img_h": img_height, "num_patches": None, "tags": [], "relations": {},
            "annotation_log": {"worker": "worker", "timestamp": None, "tool_version": "CVAT", "source": None},
            "license_tag": {"usability": True, "public": False, "commercial": True, "type": None, "holder": None}
        }

        # 텍스트 파일 파싱하여 단어 정보 추가
        ufo_image_data["words"] = parse_text_file(text_file_path)
        merged_ufo_data["images"][image_name] = ufo_image_data

    return merged_ufo_data

def save_ufo_data(ufo_data: dict, output_path: str):
    """UFO 데이터를 JSON 파일로 저장합니다."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ufo_data, f, ensure_ascii=False, indent=4)
    print(f"모든 텍스트 파일이 ufo format으로 변환, json 파일로 저장되었습니다: {output_path}")

def move_directory(src_path: str, dest_path: str):
    """디렉터리를 지정된 경로로 이동합니다."""
    try:
        shutil.copytree(src_path, dest_path)
        print(f"'{src_path}'가 '{dest_path}'로 성공적으로 이동되었습니다.")
    except Exception as e:
        print(e)
def move_file(src_path: str, dest_path: str):
    """파일을 지정된 경로로 이동합니다."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        shutil.move(src_path, dest_path)
        print(f"'{src_path}'가 '{dest_path}'로 성공적으로 이동되었습니다.")
    except Exception as e:
        print(f'json파일 이동 중 에러 :{e}')



# 경로 설정
text_files_dir = "./archive/SROIE2019/train/box"
output_json_path = "./archive/SROIE2019/train/box/train.json"
img_path = "./archive/SROIE2019/train/img"
root_dir = "./data/SROIE2019_receipt"
ann_destination_path = f"{root_dir}/ufo/train.json"
img_destination_path = f"{root_dir}/img/train"


# UFO 형식으로 변환 및 저장
ufo_data = create_ufo_data(text_files_dir, img_width=1280, img_height=1707)
save_ufo_data(ufo_data, output_json_path)

# 이미지 및 어노테이션 파일 이동
move_directory(img_path, img_destination_path)
move_file(output_json_path, ann_destination_path)
