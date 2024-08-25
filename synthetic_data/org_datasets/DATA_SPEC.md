## CDATM별거래집계 
- 설명 : CD/ATM별 거래 건수 및 금액 집계 내역
- 데이터 : [DATOP_CDATM.csv](DATOP_CDATM.csv)
  - BASE_YM : 기준연월	
  - HNDE_BANKCD : (비식별)취급은행코드	
  - TRAN_TYE_NM : 거래구분명	
  - TRAN_AMT : 거래금액(원)
- 데이터수 : 5,346개

## 타행거래집계
- 설명 : 타행간 입금(현금, 수표)거래 요약 정보
- 데이터 : [DATOP_CASH_CHECK.csv](DATOP_CASH_CHECK.csv)
  - BASE_YM : 기준연월
  - HNDE_BANKCD : (비식별)취급은행코드
  - DPOST_BANKCD : (비식별)입금은행코드
  - DPOST_TYE_NM : 입금구분명
  - AMT : 금액
- 데이터수 : 147,809개

## 타행이체거래내역
- 설명 : 전자금융홈펌뱅킹을 통해 타행으로 이체 거래내역
- 데이터 :[DATOP_HF_TRANS.csv](DATOP_HF_TRANS.csv)
  - BASE_YM : 기준연월
  - HNDE_BANK_RPTV_CODE : (비식별)취급은행대표코드
  - OPENBANK_RPTV_CODE : (비식별)개설은행대표코드
  - FND_TPCD : 자금구분코드
  - TRAN_AMT : 거래금액
- 데이터수 :  267,241개

### PG계좌이체거래내역
- 설명 : 뱅크페이 PG서비스 이용하는 가맹점 중 입금방법을 거래일 실시간/익영업일로 선택한 가맹점의 거래내역
- 데이터 : [DATOP_PG_TRANS.csv](DATOP_PG_TRANS.csv)
  - WD_YM : 출금연월
  - UTIZN_CO_APRVNO : 이용업체승인번호
  - CALT_METH_NM : 정산방법명
  - WD_BANKCD : (비식별)출금은행코드
  - DPOST_BANKCD : (비식별)입금은행코드
  - TRNSR_REQ_AMT : 이체요청금액
- 데이터수 : 438,547개