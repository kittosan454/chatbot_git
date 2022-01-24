# 추천 시스템 및 비속어 검출 챗봇
  
  
## 개요 
언어모델에 대해 `auto regressive`, `text classification` 파인튜닝 및 테스트  
- **KoGPT2**: **질의**가 주어졌을 때, 다음 **답변**에 대한 텍스트 
- **KoELECTRA**: **질의**에 대해서 **카테고리를 예측** 
- **KoBERT**:  **질의**에 대해서 **카테고리를 예측** 및 비속어 검출

## 사용 Language Model
KoBERT, KoGPT2

## 환경
### Data
- 비속어 데이터
- 챗봇 데이터
### GPU
Colab
### Package
```
kogpt2-transformers
kobert-transformers
transformers==3.0.2
torch
```

## Task
### 1. KoBERT Text Classifcation
KoBERT를 이용한 추천 시스템 구축
#### 1.1 질의에 대한 카테고리 분류
##### 데이터
-비속어 데이터 및 챗봇 대화용 데이터 사용
-추천시스템 관련 데이터 사용
  
**카테고리 클래스** 데이터: 카테고리 클래스 개

_______________________아직 미완_______________________________








## 기간
2022.01.24~2022.03.09
# References
[KoBERT](https://github.com/SKTBrain/KoBERT)  
[KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)  
[KoGPT2](https://github.com/SKT-AI/KoGPT2)  
[KoGPT2-Transformers](https://github.com/taeminlee/KoGPT2-Transformers/)  
[KoELECTRA](https://github.com/monologg/KoELECTRA)  
[enlipleai/kor_pretrain_LM](https://github.com/enlipleai/kor_pretrain_LM)  
[how-to-generate-text](https://huggingface.co/blog/how-to-generate?fbclid=IwAR2BZ4BNG0PbOvS5QaPLE0L3lx7_GOy_ePVu4X1LyTktQo-nLEPr7eht1O0)
[Korean Language Model for Wellness Conversation](https://github.com/nawnoes/WellnessConversation-LanguageModel)