# 🏝 멤버 구성 및 역할

| [전현욱](https://github.com/gusdnr122997) | [곽수연](https://github.com/gongree) | [김가영](https://github.com/garongkim) | [김신우](https://github.com/kimsw9703) | [안윤주](https://github.com/nyunzoo) |
| --- | --- | --- | --- | --- |
| <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/0a2cc555-e3fc-4fb1-9c05-4c99038603b3)" width="140px" height="140px" title="Hyunwook Jeon" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/d500e824-f86d-4e72-ba59-a21337e6b5a3)" width="140px" height="140px" title="Suyeon Kwak" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/0fb3496e-d789-4368-bbac-784aeac06c89)" width="140px" height="140px" title="Gayoung Kim" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/77b3a062-9199-4d87-8f6e-70ecf42a1df3)" width="140px" height="140px" title="Shinwoo Kim" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/f3b42c80-7b82-4fa1-923f-0f11945570e6)" width="140px" height="140px" title="Yunju An" /> |
- **전현욱**
    - 팀 리더, Soft Voting Ensemble, Reader 학습 개선, Retriever 개선
- **곽수연**
    - Context Preprocessing, Reader 학습
- **김가영**
    - Reader Negative Sampling 적용, Reader 학습
- **김신우**
    - DPR Negative Sampling 구현, Reader 학습
- **안윤주**
    - BM25 구현, DPR Bi-Encoder 학습

# 🍍 프로젝트 기간

2024.02.07 10:00 ~ 2024.02.22 19:00

# 🍌 프로젝트 소개

- Question Answering(QA)은 다양한 종류의 질문에 대답하는 인공지능을 만드는 연구 분야이다. 그중 Open-Domain Question Answering(ODQA)은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어 있는 knowledge resource에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가되어 질의에 대한 답변을 제시한다. 
- 본 프로젝트는 질문에 관련된 문서를 찾아주는 “retriever” 단계와 관련된 문서를 읽고 적절한  답변을  찾거나  만들어주는  “reader” 단계를  각각  구성하고  그것들을  적절히 통합하여 질문에 대한 답변을 해주는 ODQA 시스템을 만드는 것을 목적으로 한다.

# 🥥 프로젝트 구조

![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-09/assets/81287077/9c4c683a-d832-4787-a146-35a37e42f096)

## 데이터셋 구조

| Column | 설명 |
| --- | --- |
| id | 질문 고유의 id |
| question | 질문 |
| answers | 답변에 대한 정보. 하나의 질문에 하나의 답변만 존재함 |
| context | 답변이 포함된 문서 |
| title | 문서의 제목 |
| document_id | 문서의 고유 id |

## 평가 지표
- **Exact Match(EM)** : 모든 질문에 대하여 모델의 예측과 실제 답이 정확하게 일치할 때 1점 부여
- F1 Score : 겹치는 단어도 고려해 부분 점수 부여 (참고용)

# 🤿 사용 모델

- klue/roberta-large2 
- nlpotato/roberta_large-ssm_wiki_e2-origin_added_korquad_e53 
- uomnf97/klue-roberta-finetuned-korquad-v2

# 👒 폴더 구조

```bash
📦level2-nlp-mrc-nlp-09
 ┣ 📂code
 ┃ ┣ 📜BM25Vectorizer.py
 ┃ ┣ 📜arguments.py
 ┃ ┣ 📜baseline_train.ipynb
 ┃ ┣ 📜bm25.py
 ┃ ┣ 📜custom_retriever.py
 ┃ ┣ 📜dense_retrieval.ipynb
 ┃ ┣ 📜dpr.py
 ┃ ┣ 📜dpr_train.py
 ┃ ┣ 📜ensemble.py
 ┃ ┣ 📜inference.py
 ┃ ┣ 📜requirements.txt
 ┃ ┣ 📜retrieval.py
 ┃ ┣ 📜retrieval_bm25.py
 ┃ ┣ 📜sent_retrieval.py
 ┃ ┣ 📜sentence_trans.ipynb
 ┃ ┣ 📜train.py
 ┃ ┣ 📜trainer_qa.py
 ┃ ┣ 📜utils_qa.py
 ┃ ┗ 📜wiki_preprocess.ipynb
 ┣ 📜MRC_NLP_팀 리포트(09조).pdf
 ┗ 📜README.md
```

# 🍸 Leaderboard

|  |  Exact Match | F1 Score |
| --- | --- | --- |
| Public | 66.67 | 77.17 |
| Private | 63.67 | 74.22 |
