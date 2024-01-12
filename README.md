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

**건강 및 식음료**

- 인물: 

|Domain|Slot|Description|Example|
|---|---|---|---|
|건강 및 식음료|인물|식당과 관련된 인물(셰프), 유명 의료인의 이름|이연복, 레이먼킴 등|
||제품/서비스|제품 및 서비스의 종류 및 명칭|피자, 파스타, 치킨, 등|
||장소/조직|식당, 업체명, 매장의 명칭, 플랫폼, 병원이나 약국의 이름, 의학, 의료 관련 기관, 단체|풀무원, 대웅제약, 성모병원, 국민은행 등|
||지역|식당이나 의료 시설과 관련된 지역의 지명|연희동, 일원동, 속초 등|
||영양소/성분|||

### 1.3. 다운로드



## 2. 사용법

```python

from transformers import AutoTokenizer

```
