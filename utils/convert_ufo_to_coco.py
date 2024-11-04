def convert_from_ufo_to_coco_format(data):
    '''
    input : UFO 형식 json file
    output : COCO 형식 json file
    '''
    image_id_start = 1
    annotation_id_start = 1

    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 0, 'name': 'text'}]
    }

    for file_name, file_info in data['images'].items():
        image_id = image_id_start
        coco_image = {
            'width': file_info['img_w'],
            'height': file_info['img_h'],
            'file_name': file_name,
            'license': 0,
            'flickr_url': None,
            'coco_url': None,
            'date_captured': "2024-05-30",
            'id': image_id
        }
        coco_data['images'].append(coco_image)

        for word_id, word_info in file_info['words'].items():
            annotation_id = annotation_id_start
            [tl, tr, br, bl] = word_info['points']
            width = max(tl[0], tr[0], br[0], bl[0]) - min(tl[0], tr[0], br[0], bl[0])
            height = max(tl[1], tr[1], br[1], bl[1]) - min(tl[1], tr[1], br[1], bl[1])
            min_x = min(tl[0], tr[0], br[0], bl[0])
            min_y = min(tl[1], tr[1], br[1], bl[1])
            coco_annotation = {
                'image_id': image_id,
                'category_id': 0, 
                'area': width*height,
                'bbox': [min_x, min_y, width, height],
                'iscrowd': 0,
                'id': annotation_id
            }
            coco_data['annotations'].append(coco_annotation)

            annotation_id_start += 1
        image_id_start += 1
    
    return coco_data