# 한국어 SNS 멀티턴 대화 데이터

**한국어 SNS 멀티턴 대화 데이터** 프로젝트는 한국어 기반의 SNS 스타일 대화 데이터셋을 구축하기 위해 진행된 프로젝트입니다. 해당 저장소는 프로젝트 진행 과정에서 유효성 검증을 위해 사용한 모델 공유를 위해 생성되었습니다.

## 1. 개요

**한국어 SNS 멀티턴 대화 데이터**의 도메인은 총 9개로 나누어져있으며, 일상 주제 6개와 시사 주제 3개로 구분됩니다.

|Category|Domain|Description|
|:---:|---:|:---|
|일상|건강 및 식음료|우리의 몸이나 정신의 상태를 가꿔서 나아지거나 나빠지는 것과 관련한 이야기, 식음료와 관련한 배달, 외식, 식당 조리 등을 다루는 이야기|
|일상|여행, 관광 및 명소|다녀왔거나 예정된 여행지 및 명소에 관련된 이야기|
|일상|문화 생활 및 여가|문화 생활이나 취미 활동 등 여유 시간을 즐기기 위한 행위나 활동에 관련된 이야기|
|일상|미용과 패션|미용 및 스타일에 관련된 이야기, 의류와 악세사리, 가방 등의 소품 전반을 다룸|
|일상|스포츠 및 e스포츠|스포츠나 E-스포츠 대회에 대한 감상과 관련된 이야기, 경기 결과나 팀, 선수에 대한 일반적인 사실, 평가 전반을 다룸|
|일상|콘텐츠 소비|제작된 방송 및 영상 컨텐츠에 대한 감상과 관련된 이야기|
|시사|정치|정치 현안 및 제도에 관련된 이야기, 편향적이거나 혐오가 담긴 발화가 포함되지 않도록 유의하였음|
|시사|경제 및 사회|사회 현상 및 이슈를 다루는 이야기, 경제 활동과 관련된 제도, 정책에 관한 이야기|
|시사|과학 기술|과학 기술과 그에 관련된 이슈를 다루는 이야기, 과학 기술을 이용한 기기나 장치, 프로그램을 다룸|

### 1.1. 데이터 수량

|Domain|Num of Dialogs|Ratio|
|---:|:---:|:---|
|건강 및 식음료|19,102|10.4%|
|여행, 관광 및 명소|22,800|12.4%|
|문화 생활 및 여가|19,888|10.8%|
|미용과 패션|20,601|11.2%|
|스포츠 및 e스포츠|20,291|11.0%|
|콘텐츠 소비|19,076|10.3%|
|정치|2,718|1.5%|
|경제 및 사회|41,056|22.3%|
|과학 기술|18,947|10.3%|

### 1.2. 주제별 슬롯

파일 참조: [주제별 슬롯.pdf](https://github.com/trailerAI/Multi_Turn_Modeling/files/13934829/slot.pdf)

### 1.3. 다운로드

추가 예정.

## 2. 사용법

### 2-1. 환경 구축

```shell

# Docker Container 빌드
docker import ./multi_turn.tar multi_turn_model:latest

# Docker Container 생성 및 실행
docker run --ipc=host -it --name multi_turn_container --gpus all -p 5103:5103 multi_turn_model /bin/bash

# 한국어 패키지 빌드
export LANGUAGE=ko && apt=get update && apt-get install locales
localedef -f UTF-8 -i ko_KR ko_KR.UTF-8
locale-gen ko_KR.UTF-8 && export LC_ALL=ko_KR.UTF-8
LC_ALL=ko_KR.UTF-8 bash

# 학습 모델 경로 이동
cd /home/data

```

### 2-2. 데이터 다운로드

데이터를 다운로드하여 Multi_Turn_Modeling 폴더에 복사합니다.

```
Multi_Turn_Modeling/
├── Training/
|   ├── 건강_및_식음료.json
|   ├── 여행_관광_및_명소.json
|   ├── ...
|   └── 과학_기술.json
├── Validation/
|   └── ...
├── Test/
|   └── ...
├── Others/
|   └── ...
├── dataset.py
├── model.py
├── README.md
├── test.py
├── train.py
└── utils.py
```

### 2-3. 학습 및 검증

```shell

python train.py -td data/건강_및_식음료/train.json \
-vd data/건강_및_식음료/eval.json \
-od data/건강_및_식음료/ontology.json

```

**사용 가능한 파라미터**

```
-td, --train_dir: 학습 데이터 경로 (필수)
-vd, --valid_dir: 검증 데이터 경로 (필수)
-od, --ontology_dir: ontology 경로 (필수)
-pm, --pretrained_model: 사전학습된 모델 경로, huggingface 경로와 file system 경로 모두 가능 (default: yeongjoon/kconvo-roberta)
-e, --num_epochs: 에폭 수 (default: 100)
-p, --patience: early stopping patience. 미사용시 -1 (default: 3)
-b, --batch_size: 배치 크기, 본인 메모리에 맞게 2의 지수배로 입력 (default: 64)
-ml, --max_length: 입력 최대 길이, 변경 x (default: 512)
-lr, --learning_rate: 학습률, AdaGrad 논문 참조 (default: 5e-5)
```

### 2-4. 시험

```shell

python test.py -d data/건강_및_식음료/test.json \
-od data/건강_및_식음료/ontology.json

```

**사용 가능한 파라미터**

```
-d, --data_dir: 시험 데이터 경로 (필수)
-od, --ontology_dir: ontology 경로 (필수)
-sd, --save_dir: 파일 저장 경로 (default: ./result_test/)
-b, --batch_size: 배치 크기, 본인 메모리에 맞게 2의 지수배로 입력 (default: 64)
-ml, --max_length: 입력 최대 길이, 변경 x (default: 512)
```

시험을 마치면 저장 경로에 성능 측정 결과인 performance_{domain}.txt, 예측 결과인 pred_{domain}.txt 파일이 생성됩니다.

## 3. 모델

Encoder Based Transformer 구조를 활용하여 모델을 구축하였습니다. BERT, RoBERTa, ELECTRA 등의 후보 모델을 예비실험한 결과, RoBERTa를 최종 선정하였습니다.

### 3-1. RoBERTa

<img src="https://github.com/trailerAI/Multi_Turn_Modeling/assets/45366231/ab7d3c15-a568-4b04-9ef0-999af4fd33f2" width=50% height=50%>

Endoer-Based Transformer 구조를 활용하여 모델을 구축하였습니다.

### 3-2. Token Classification

<img src="https://github.com/trailerAI/Multi_Turn_Modeling/assets/45366231/07ed8195-062f-46eb-a644-74827305ac1a" width=50% height=50%>

Token 단위에서 분류하는 Token Classification 기법을 사용하여 예측하였습니다.
<CLS> 토큰에서 특수 슬롯(가격대, 평가/후기/감상)의 5가지 상태(yes, no, soso, dontcare, none)를 예측하고, 나머지 토큰에서 NER과 같은 BIO 레이블링을 통해 Dialog State를 예측합니다.

## 4. 실험

### 4-1. 환경

실험 환경은 아래와 같으며, **2-1 환경 구축** 단계를 거칠 경우 하드웨어를 제외한 모든 환경을 동일하게 구축할 수 있습니다.

```
CPU: Intel(R) Xeon(R) Silver 4215 CPU @ 2.50GHz
GPU: NVIDIA GeForce RTX 3090 * 4
RAM: 512GB

OS: Ubuntu 18.04.6 LTS
Frameworks:
Python 3.6.9
PyTorch 1.8.0+cu111
```

### 4-2. 사용 파라미터

**학습**

```shell
python train.py -td ./Training/{domain}.json \
-vd ./Validation/{domain}.json \
-od ./Others/{domain}_ontology.json \
-pm yeongjoon/kconvo-roberta \
-e 100 \
-p 3 \
-b 64 \
-ml 512 \
-lr 0.00005
```

**시험**

```shell
python test.py -d ./Test/{domain}.json \
-od ./Others/{domain}_ontology.json \
-sd ./result_test/ \
-b 64 \
-ml 512
```

### 4-3. 성능

성능 측정은 KLUE Benchmark에서 DST 성능 측정을 위해 사용하는 Slot-F1 소스 코드를 사용하였으며, Precision, Recall과 TP, FP, FN을 출력하기 위해 약간의 수정을 하였습니다.
수정된 코드는 [utils.py](https://github.com/trailerAI/Multi_Turn_Modeling/blob/main/utils.py)의 compute_f1 메소드를 확인하시면 됩니다.

|도메인|Precision|Recall|F1|
|---:|---|---|---|
|건강 및 식음료|0.940934066|0.911690996|0.926081735|
|여행, 관광 및 명소|0.926778489|0.912496563|0.919582077|
|문화 생활 및 여가|0.93733292|0.903322985|0.920013751|
|미용과 패션|0.925879421|0.91744191|0.921641355|
|스포츠 및 e스포츠|0.916507689|0.932449188|0.924409716|
|콘텐츠 소비|0.918865119|0.930325141|0.924559619|
|정치|0.932717949|0.926522129|0.929609715|
|경제 및 사회|0.958449192|0.901497959|0.929101657|
|과학 기술|0.938801403|0.897130274|0.917492923|

실험 결과 평균 F1은 0.9236으로 유효성 평가 기준인 0.92를 통과하였으며, **여행, 관광 및 명소**, **과학 기술**을 제외한 모든 도메인에서 0.92 이상의 성능을 달성하였습니다.

## 5. 향후 연구

### 5-1. Task Oriented Dialog System

<img src="https://github.com/trailerAI/Multi_Turn_Modeling/assets/45366231/222fd8f8-8d93-46f5-aa71-876c74b718c8" width=50% height=50%>

대화 상태 추적은 일반적으로 목적 지향 대화 시스템의 하위 시스템으로서 동작하며, 이를 이용하여 1. Detection 2. Selection 3. Generation 단계를 거치는 End-to-End 대화 시스템을 구축할 계획입니다.
이를 위해 자체적으로 Knowledge Base를 구축하고, Text Generation 모델을 설계할 계획입니다.

### 5-2. LLM

LLM 모델을 학습시키기 위해서 여러 도메인과 다양한 태스크의 데이터셋이 필요합니다. 해당 데이터셋은 다양한 도메인을 만족하고 있으며, 타 태스크의 데이터셋과 함께 사용한다면 더욱 강건한 모델을 구축할 수 있습니다.
추가로, RLHF와 DPO, PPO 등의 강화학습 기법을 사용한다면 LLM 학습이 가능합니다.

![ML-14874_image001](https://github.com/trailerAI/Multi_Turn_Modeling/assets/45366231/de01e125-f257-41b9-b606-46704b20e640)


**이 프로젝트는 2023년도 정부(과학기술정보통신부)의 재원으로 한국지능정보사회진흥원의 지원을 받아 수행된 프로젝트입니다.**
