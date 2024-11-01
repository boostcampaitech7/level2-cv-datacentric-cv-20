## Streamlit 기능

1. train data에 train.json 으로 시각화 기능

2. test data에 리더보드 제출한 csv 파일로 시각화 기능   



## Streamlit 사용법

1. Streamlit으로 디렉토리 변경
```
cd Streamlit
```

2. `streamlit.py` 실행
```
streamlit run streamlit.py --server.runOnSave=true
```



## 프로젝트 구조
```
├─ dataset
│   ├─ train
│   ├─ test
│   ├─ train.json
│   └─ test.json
│
├─ Streamlit
│   ├─ output
│   ├─ streamlit.py
│   └─ README.md

```