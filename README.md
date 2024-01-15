# 한국어 SNS 멀티턴 대화 데이터

**한국어 SNS 멀티턴 대화 데이터** 프로젝트는 한국어 기반의 SNS 스타일 대화 데이터셋을 구축하기 위해 진행된 프로젝트입니다. 해당 저장소는 프로젝트 진행 과정에서 유효성 검증을 위해 사용한 모델 공유를 위해 생성되었습니다.

## 1. 개요

**한국어 SNS 멀티턴 대화 데이터**의 도메인은 총 9개로 나누어져있으며, 일상 주제 6개와 시사 주제 3개로 구분됩니다.

|Category|Domain|Description|
|:---:|:---:|:---|
|일상|건강 및 식음료|우리의 몸이나 정신의 상태를 가꿔서 나아지거나 나빠지는 것과 관련한 이야기, 식음료와 관련한 배달, 외식, 식당 조리 등을 다루는 이야기|
|일상|여행, 관광 및 명소|다녀왔거나 예정된 여행지 및 명소에 관련된 이야기|
|일상|문화 생활 및 여가|문화 생활이나 취미 활동 등 여유 시간을 즐기기 위한 행위나 활동에 관련된 이야기|
|일상|미용과 패션|미용 및 스타일에 관련된 이야기, 의류와 악세사리, 가방 등의 소품 전반을 다룸|
|일상|스포츠 및 E-스포츠|스포츠나 E-스포츠 대회에 대한 감상과 관련된 이야기, 경기 결과나 팀, 선수에 대한 일반적인 사실, 평가 전반을 다룸|
|일상|방송 연예 및 콘텐츠 소비|제작된 방송 및 영상 컨텐츠에 대한 감상과 관련된 이야기|
|시사|사회 및 경제|사회 현상 및 이슈를 다루는 이야기, 경제 활동과 관련된 제도, 정책에 관한 이야기|
|시사|정치 및 정세|정치 현안 및 제도에 관련된 이야기, 편향적이거나 혐오가 담긴 발화가 포함되지 않도록 유의하였음|
|시사|IT, 과학|과학 기술과 그에 관련된 이슈를 다루는 이야기, 과학 기술을 이용한 기기나 장치, 프로그램을 다룸|

### 1.1. 데이터 수량

|Domain|Num of Dialogs|Num of Utterances|Ratio|
|---|---|---|---|
|건강 및 식음료|-|-|10%|
|여행, 관광 및 명소|-|-|10%|
|문화 생활 및 여가|-|-|10%|
|미용과 패션|-|-|10%|
|스포츠 및 E-스포츠|-|-|10%|
|방송 연예 및 콘텐츠 소비|-|-|10%|
|사회 및 경제|-|-|25%|
|정치 및 정세|-|-|5%|
|IT, 과학|-|-|10%|

### 1.2. 주제별 슬롯

파일 참조: [주제별 슬롯.pdf](https://github.com/trailerAI/Multi_Turn_Modeling/files/13934829/slot.pdf)

### 1.3. 다운로드

추가 예정.

## 2. 사용법

### 2-1. 저장소 다운로드

```shell

git clone https://github.com/trailerAI/Multi_Turn_Modeling.git
cd Multi_Turn_Modeling

```

### 2-2. 데이터 다운로드

```
Multi_Turn_Modeling/
├── data/
|   ├── 건강_및_식음료/
|   |   ├── train.json
|   |   ├── eval.json
|   |   ├── test.json
|   |   └── ontology.json
|   ├── 여행_관광_및_명소/
|   ├── ...
|   └── IT_과학/
├── dataset.py
├── model.py
├── README.md
├── test.py
├── train.py
└── utils.py
```

### 2-3. 학습

```shell

python train.py -td data/건강_및_식음료/train.json \
-vd data/건강_및_식음료/eval.json \
-od data/건강_및_식음료/ontology.json

```

**사용 가능한 파라미터**

```
-td, --train_dir: 학습 데이터 경로 (필수)
-vd, --eval_dir: 검증 데이터 경로 (필수)
-od, --ontology_dir: ontology 경로 (필수)
-pt, --pretrained_tokenizer: 사전학습된 토크나이저 경로, huggingface 경로와 file system 경로 모두 가능 (default: yeongjoon/kconvo-roberta)
-pm, --pretrained_model: 사전학습된 모델 경로, huggingface 경로와 file system 경로 모두 가능 (default: yeongjoon/kconvo-roberta)
-e, --num_epochs: 에폭 수 (default: 100)
-p, --patience: early stopping patience. 미사용시 -1 (default: 3)
-b, --batch_size: 배치 크기, 본인 메모리에 따라 2의 지수배로 입력 (default: 64)
-ml, --max_length: 입력 최대 길이, 변경 x (default: 512)
```


