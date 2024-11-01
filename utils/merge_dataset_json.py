import os
import shutil
import json

def merge_images_and_json_files(data_path, new_data_path):
    """
    기존 언어별로 나뉘어진 데이터와 json 파일을 통합하여 하나의 train/ test 폴더에 저장하는 함수

    Parameters:
        data_path (str): 원본 데이터 폴더 경로. "../data"
        new_data_path (str): 통합된 데이터가 저장될 폴더 경로. "../dataset"
    """
    # 새로운 통합 폴더 경로 설정
    output_train_path = os.path.join(new_data_path, "train")
    output_test_path = os.path.join(new_data_path, "test")

    # 통합 폴더가 없다면 생성
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    # 빈 딕셔너리 생성
    merged_train_data = {"images": {}}
    merged_test_data = {"images": {}}


    # 데이터 폴더 리스트
    folders = ["chinese_receipt", "japanese_receipt", "thai_receipt", "vietnamese_receipt"]

    # 모든 폴더의 train, test 이미지 파일 복사
    for folder in folders:
        # 각 폴더의 train/test 경로
        train_folder = os.path.join(data_path, folder, "img", "train")
        test_folder = os.path.join(data_path, folder, "img", "test")
        
        # train 이미지 복사
        if os.path.exists(train_folder):
            for img_file in os.listdir(train_folder):
                src = os.path.join(train_folder, img_file)
                dst = os.path.join(output_train_path, img_file)
                shutil.copy2(src, dst)
        
        # test 이미지 복사
        if os.path.exists(test_folder):
            for img_file in os.listdir(test_folder):
                src = os.path.join(test_folder, img_file)
                dst = os.path.join(output_test_path, img_file)
                shutil.copy2(src, dst)

        # 각 폴더의 train/test JSON 파일 경로
        train_json_path = os.path.join(data_path, folder, "ufo", "train.json")
        test_json_path = os.path.join(data_path, folder, "ufo", "test.json")
        
        # train JSON 병합
        if os.path.exists(train_json_path):
            with open(train_json_path, "r") as f:
                train_data = json.load(f)
                # images 키의 데이터 병합 (폴더명 접두사 추가하여 고유 유지)
                for key, image_data in train_data["images"].items():
                    merged_train_data["images"][key] = image_data

        # test JSON 병합
        if os.path.exists(test_json_path):
            with open(test_json_path, "r") as f:
                test_data = json.load(f)
                # images 키의 데이터 병합 (폴더명 접두사 추가하여 고유 유지)
                for key, image_data in test_data["images"].items():
                    merged_test_data["images"][key] = image_data

    # 최종 병합 JSON 파일 저장
    with open(os.path.join(new_data_path, "train.json"), "w") as f:
        json.dump(merged_train_data, f, indent=4)

    with open(os.path.join(new_data_path, "test.json"), "w") as f:
        json.dump(merged_test_data, f, indent=4)

    # 데이터가 잘 병합되었는지 확인
    if len(os.listdir(output_train_path)) == 400 and len(merged_train_data['images'].keys()) == 400:
            print("Train DATA and JSON merged successfully!")
    if len(os.listdir(output_test_path)) == 120 and len(merged_test_data['images'].keys()) == 120:
            print("Test DATA and JSON merged successfully!")
    

data_path = "./data"
new_data_path = "./dataset"
merge_images_and_json_files(data_path, new_data_path)
