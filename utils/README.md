## utils 내부 파일 메뉴얼

1. `convert_ufo_to_coco.py` : UFO JSON 파일에서 COCO format JSON 파일로 변환
```bash
python convert_ufo_to_coco.py --input_path path/to/input_ufo.json --output_path path/to/output_coco.json
```

2. `download_CORD_dataset.py` : `./data/`  경로에 CORD 데이터셋 다운로드
```bash
python download_CORD_dataset.py
```

3. `merge_dataset.py` : 이미지와 JSON 파일을 통합하여 하나의 train/test 폴더로 병합
```bash
python merge_dataset.py --data_path path/to/data --new_data_path path/to/dataset
```

4. `split_data.py` : `./data/`  경로에 있는 기존 데이터셋을 훈련, 검증 데이터셋으로 나눈 json 파일 생성
```bash
python split_data.py
```

5. `convert_image_crop.py` : `./data/`  
```bash
python convert_image_crop.py
# parser.add_argument('--base_dir', type=str, default=default_data_dir)
# parser.add_argument('--output_dir', type=str, default="cropped_data")
# parser.add_argument('--padding', type=int, default=50)
# parser.add_argument('--val', type=bool, default=False)
```

6. `convert_image_to_gray.py` : `./data/`  경로에 있는 기존 데이터셋을 훈련, 검증 데이터셋으로 나눈 json 파일 생성
```bash
python convert_image_to_gray.py
# parser.add_argument('--base_dir', type=str, default=default_data_dir)
# parser.add_argument('--output_dir', type=str, default="gray_data")
# parser.add_argument('--val', type=bool, default=False)
```

7. `convert_image_to_preprocessing.py` : `./data/`  경로에 있는 기존 데이터셋을 훈련, 검증 데이터셋으로 나눈 json 파일 생성
```bash
python convert_image_to_preprocessing.py
# parser.add_argument('--base_dir', type=str, default=default_data_dir)
# parser.add_argument('--output_dir', type=str, default="prep_data")
# parser.add_argument('--val', type=bool, default=False)
```

8. `make_bottom_n%_dataset.py` : `./data/`  경로에 있는 기존 데이터셋을 훈련, 검증 데이터셋으로 나눈 json 파일 생성
```bash
python make_bottom_n%_dataset.py
# parser.add_argument('--base_dir', type=str, default=default_data_dir)
# parser.add_argument('--output_dir', type=str, default="bottom_iou_data")
# parser.add_argument('--gt_path', type=str, default=default_data_dir)
# parser.add_argument('--pred_path', type=str, default=script_dir+"/pred_output.json")
# parser.add_argument('--iou_path', type=str, default=script_dir+"/iou_data.json")
# parser.add_argument('--is_compare_data', type=bool, default=False)
# parser.add_argument('--iou', type=int, default=30)
# parser.add_argument('--val', type=bool, default=False)
```