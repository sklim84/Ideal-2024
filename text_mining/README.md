## 1. 키워드 분석 
- 사용기법 : [KeyBert](https://maartengr.github.io/KeyBERT/api/keybert.html)
- 분석내용
  - 각 컬럼 내 텍스트를 기반으로 단어의 중요도 분석
  - SUSI 데이터는 그룹별/컬럼별 텍스트 기반 키워드 분석 수행
  - 키워드 분석 수행 결과는 주요 키워드와 스코어로 구성
- 소스코드
  - keyword.py : [BIZ_CONSULTING_RESULT_202411101625.xlsx](datasets%2FBIZ_CONSULTING_RESULT_202411101625.xlsx) 분석
  - keyword_susi.py : [BIZ_CONSULTING_SUSI_RESULT_NEW_202411101627_pp.xlsx](datasets%2FBIZ_CONSULTING_SUSI_RESULT_NEW_202411101627_pp.xlsx) 분석
- 분석결과 : ./results/keyword
  - BIZ_CON_RES : [BIZ_CONSULTING_RESULT_202411101625.xlsx](datasets%2FBIZ_CONSULTING_RESULT_202411101625.xlsx) 분석결과
  - BIS_CON_RES_SUSI : [BIZ_CONSULTING_SUSI_RESULT_NEW_202411101627_pp.xlsx](datasets%2FBIZ_CONSULTING_SUSI_RESULT_NEW_202411101627_pp.xlsx) 분석

## 2. 토픽 모델링
- 사용기법 : [BERTopic](https://maartengr.github.io/BERTopic/index.html) 
- 분석내용
  - 각 컬럼 내 텍스트를 기반으로 토픽 모델링 수행
  - SUSI 데이터는 그룹별/컬럼별 텍스트 기반 토픽 모델링 수행 
  - 토픽 모델링 수행 결과는 토픽 번호와 토픽의 주요 키워드들로 구성
- 소스코드
  - topic.py : [BIZ_CONSULTING_RESULT_202411101625.xlsx](datasets%2FBIZ_CONSULTING_RESULT_202411101625.xlsx) 분석
  - topic_susi.py : [BIZ_CONSULTING_SUSI_RESULT_NEW_202411101627_pp.xlsx](datasets%2FBIZ_CONSULTING_SUSI_RESULT_NEW_202411101627_pp.xlsx) 분석
- 분석결과 : ./results/topic
  - BIZ_CON_RES : [BIZ_CONSULTING_RESULT_202411101625.xlsx](datasets%2FBIZ_CONSULTING_RESULT_202411101625.xlsx) 분석결과
  - BIS_CON_RES_SUSI : [BIZ_CONSULTING_SUSI_RESULT_NEW_202411101627_pp.xlsx](datasets%2FBIZ_CONSULTING_SUSI_RESULT_NEW_202411101627_pp.xlsx) 분석

## 3. 워드 클라우드
- 사용기법 : [WordCloud](https://amueller.github.io/word_cloud/)
- 소스코드 : 키워드 분석/토픽 모델링 소스코드 내 포함
- 분석결과 : ./results/keyword or topic/BIZ_CON_RES or BIS_CON_RES_SUSI/wordcloud
