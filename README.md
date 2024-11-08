## 1. 📖 프로젝트 개요

### 1.1 프로젝트 소개

이 프로젝트의 주제는 **Data-Centric 접근을 통한 다국어 영수증 텍스트 검출**입니다. OCR (Optical Character Recognition) 기술은 이미지 속의 텍스트를 컴퓨터가 인식할 수 있도록 변환하는 기술로, 이번 프로젝트에서는 다양한 언어의 영수증 데이터를 대상으로 OCR 모델을 학습시키고 최적화하는 데 중점을 둡니다. 

따라서 저희는 **Data-Centric 접근**을 통해 학습 데이터를 보강하고 정제하여 모델의 성능을 향상시키는 전략을 채택하였으며, 이를 통해 OCR 모델이 언어에 상관없이 영수증의 텍스트를 빠르고 정확하게 검출할 수 있도록 목표를 설정했습니다.

프로젝트 기간 : 24.09.30 ~ 24.10.28

### 1.2 학습 환경 구성

- **GPU:** Tesla V100-SXM2-32GB
- **CPU**: Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- **운영체제**: Ubuntu 20.04.6 LTS (Focal Fossa)



```
부스트코스 강의 수강 및 과제 : 24.09.30 ~ 24.10.06
데이터 EDA / 데이터 전처리 / 베이스라인 모델 학습 : 24.10.07 ~ 24.10.13
데이터 증강 및 모델 성능 개선 : 24.10.14 ~ 24.10.18
하이퍼 파라미터 튜닝 / 앙상블 : 24.10.19 ~ 24.10.24
최종 자료 정리 및 문서화 : 24.10.25 ~ 24.10.28
```

<br/>
<br/>

## 2.🧑‍🤝‍🧑 Team ( CV-20 : CV Up!!)

<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/kaeh3403"><img height="110px"  src="https://avatars.githubusercontent.com/kaeh3403"></a>
            <br/>
            <a href="https://github.com/kaeh3403"><strong>김성규</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/sweetpotato15"><img height="110px"  src="https://avatars.githubusercontent.com/sweetpotato15"/></a>
            <br/>
            <a href="https://github.com/sweetpotato15"><strong>김유경</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/jeajin"><img height="110px"  src="https://avatars.githubusercontent.com/jeajin"/></a>
            <br/>
            <a href="https://github.com/jeajin"><strong>김재진</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/SuyoungPark11"><img height="110px" src="https://avatars.githubusercontent.com/SuyoungPark11"/></a>
            <br />
            <a href="https://github.com/SuyoungPark11"><strong>박수영</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/uddaniiii"><img height="110px" src="https://avatars.githubusercontent.com/uddaniiii"/></a>
            <br />
            <a href="https://github.com/uddaniiii"><strong>이단유</strong></a>
            <br />
        </td>
</table> 

|Name|Roles|
|:----------:|:------------------------------------------------------------:|
|김성규| 타임라인 관리, EDA 분석 |
|김유경| 보고서 관리, slack 아이디어 공유, EDA 분석 |
|김재진| Github 관리, EDA 분석 |
|박수영| 코드 리팩토링, EDA 분석|
|이단유| 보고서 관리, EDA 분석|

wrap up 레포트 : [wrap up report](https://github.com/boostcampaitech7/level2-objectdetection-cv-20/blob/main/Object%20Det_CV_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(20%EC%A1%B0).pdf)

<br/>
<br/>

## 3. 💻 프로젝트 수행 

<br/>
<br/>

## 4

<br/>
<br/>

## 5. 프로젝트 구조
프로젝트는 다음과 같은 구조로 구성되어 있습니다. 
```
📦level2-cv-datacentric-cv-20
 ┣ 📂base
 ┣ 📂data
 ┣ 📂data_loader
 ┣ 📂eda
 ┣ 📂predictions
 ┣ 📂trained_models
 ┣ 📜deteval.py
 ┣ 📜inference.py
 ┣ 📜requirements.txt
 ┗ 📜train.py
```

<br/>


## 6. 기타사항

- 본 프로젝트에서 사용한 데이터셋의 적용 저작권 라이선스인 CC-BY-NC-ND([link](https://creativecommons.org/licenses/by/2.0/kr/))의 가이드를 준수하고 있습니다.

