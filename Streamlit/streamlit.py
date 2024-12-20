import streamlit as st
import os
import json
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

def load_json(json_path):
    '''
    json 파일 읽어오는 함수
    '''
    with open(json_path, 'r') as data:
        json_data = json.load(data)
    return json_data

def visualize_image_with_bboxes(image_path, points, width=3):
    '''
    이미지에 바운딩 박스를 그리는 함수
    input - image_path : 이미지 파일 경로 (../dataset/train/extractor.zh.in_house.appen_000767_page0001.jpg)
          - points : (해당 이미지의 points -> data['images']['extractor.zh.in_house.appen_001017_page0001.jpg']['words'])
          - width : bbox 선 굵기
    output - bbox 그려진 img
    '''
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert("RGB")  # EXIF 데이터를 기반으로 이미지 방향 고정

        draw = ImageDraw.Draw(img)
        
        for obj_k, obj_v in points.items():
            pts = [(int(p[0]), int(p[1])) for p in obj_v['points']]
            pt1 = sorted(pts, key=lambda x: (x[1], x[0]))[0]  # 바운딩 박스의 상단 좌표
            
            # 바운딩 박스와 텍스트 그리기
            draw.polygon(pts, outline=(255, 0, 0), width=width)
            draw.text((pt1[0] - 3, pt1[1] - 12), obj_k, fill=(0, 0, 0))
        
        return img
    
    except Exception as e:
        st.error(f"이미지 시각화 중 오류 발생: {e}")
        return None

def show_image_to_streamlit(image_folder_path, json_path, anno=True):
    '''
    3*3으로 Streamlit 페이지에 이미지 표시
    input - image_folder_path : 이미지가 존재하는 폴더 경로
           - json_path : 해당 이미지 json 파일 경로
           - anno : 바운딩 박스를 표시할지 여부 (True/False)
    '''
    json_data = load_json(json_path)

    if json_data is not None:
        images_per_page = 9
        image_ids = list(json_data['images'].keys())
        num_pages = (len(image_ids) + images_per_page - 1) // images_per_page
        page = st.slider("페이지 선택", 1, num_pages, 1) - 1

        # 현재 페이지에 표시할 이미지 ID
        start_idx = page * images_per_page
        end_idx = start_idx + images_per_page
        cols = st.columns(3)  # 3열로 구성

        for idx, image_id in enumerate(image_ids[start_idx:end_idx]):
            image_path = os.path.join(image_folder_path, image_id)

            if anno:  # 바운딩 박스를 표시할 경우
                image_annotations = json_data['images'][image_id]["words"]
                # 바운딩 박스가 표시된 이미지 생성
                img_with_bboxes = visualize_image_with_bboxes(image_path, image_annotations)
            else:  # 바운딩 박스를 표시하지 않을 경우
                img_with_bboxes = Image.open(image_path).convert("RGB")  # 이미지 로드

            # Streamlit에 이미지와 파일 이름 표시
            with cols[idx % 3]:  # 현재 열에 이미지 배치
                if img_with_bboxes:
                    st.image(img_with_bboxes, caption=image_id, use_column_width=True)


def csv_list(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    return csv_files

def check_same_csv(name, csv):
    i = 1
    while name in csv:
        if i == 1:
            name = name[:-4]+'_'+str(i)+'.csv'
        else:
            name = name[:-6]+'_'+str(i)+'.csv'
        i += 1
    return name

@st.dialog("csv upload")
def upload_csv(csv):
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    # 파일이 업로드되면 처리
    if uploaded_file is not None:
        # Pandas를 사용해 CSV 파일 읽기
        df = pd.read_csv(uploaded_file)

        # DataFrame 내용 출력
        st.write("Data Preview:")
        st.dataframe(df)

        input_name = st.text_input("csv 파일 이름 지정", value=uploaded_file.name.replace('.csv', ''))
        if st.button("upload_csv"):
            name = check_same_csv(input_name+'.csv',csv)
            st.write("saved file name: "+name)
            df.to_csv('./output/'+name,index=False)
        if st.button("close"):
            st.rerun()


# Streamlit 
def main():
    st.set_page_config(page_title='Receipt 데이터셋 이미지 확인', layout='wide')

    if st.sidebar.button("새로고침"):
        st.rerun()    
    st.title("Receipt JSON Bbox 시각화")

    option = st.sidebar.selectbox("데이터 선택",("이미지 데이터"))
    
    if option == "이미지 데이터" : 
        # 트레인 데이터 출력
        choose_data = st.sidebar.selectbox("Train, Valid, Test, Cord", ("train", "valid", "test", "cord"))
    
        if choose_data == 'train':
            st.header("Train 데이터")

            image_folder_path = '../dataset/train'
            json_path = '../dataset/train.json'

            show_image_to_streamlit(image_folder_path, json_path)
            
        elif choose_data == 'cord':
            st.header("Cord 데이터")

            image_folder_path = '../CORD_dataset/train'
            json_path = '../CORD_dataset/train.json'

            show_image_to_streamlit(image_folder_path, json_path)


        elif choose_data == 'valid':
            st.header("Valid 데이터")
            st.write('추후 수정 예정')

        else:
            st.header("Test 데이터")
            dir = 'output'
            csv = csv_list(dir)

            choose_csv = st.sidebar.selectbox("output.csv적용",("안함",)+tuple(csv))

            if st.sidebar.button("새 csv 파일 업로드"):
                upload_csv(csv)

            if choose_csv != "안함":
                image_folder_path = '../dataset/test'
                json_path = os.path.join(dir, choose_csv)

                show_image_to_streamlit(image_folder_path, json_path)

            else:
                image_folder_path = '../dataset/test'
                json_path = '../dataset/test.json'
                
                show_image_to_streamlit(image_folder_path, json_path, anno=False)

if __name__ == "__main__":
    main()
