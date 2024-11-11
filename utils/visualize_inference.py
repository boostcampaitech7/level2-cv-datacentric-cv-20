import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


json_path = "../predictions/remove_sc/output.json"

def visualize_bounding_boxes(image_path, annotations):
    if not os.path.exists(image_path):
        return

    image = cv2.imread(image_path)
    if image is None:
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for bbox in annotations:
        points = bbox['points']
        cv2.polylines(
            image_rgb, [np.array(points, np.int32).reshape((-1, 1, 2))],
            isClosed=True, color=(0, 255, 0), thickness=2
        )

    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

image_paths = [
    "../data/chinese_receipt/img/test",
    "../data/japanese_receipt/img/test",
    "../data/thai_receipt/img/test",
    "../data/vietnamese_receipt/img/test"
]

for image_name, image_info in data['images'].items():
    image_path = None
    for path in image_paths:
        potential_image_path = os.path.join(path, image_name)
        if os.path.exists(potential_image_path):
            image_path = potential_image_path
            break
            
    if image_path is None:
        continue  
    
    annotations = []
    for word_data in image_info['words'].values():
        annotations.append({
            "points": word_data['points']
        })

    if not annotations:
        continue

    visualize_bounding_boxes(image_path, annotations)
