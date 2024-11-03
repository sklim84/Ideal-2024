import pandas as pd
from sdmetrics.single_table import CategoricalCAP

import warnings

warnings.filterwarnings("ignore")

# 원본 데이터 로드
original_data = pd.read_csv("../synthetic_data/org_datasets/DATOP_HF_TRANS_ENC_CODE.csv")
#original_data = original_data[[col for col in original_data.columns if col != "HNDE_BANK_RPTV_CODE"]]

# 세 개의 합성 데이터 로드
synthetic_data_files = ["data/synthetic_data_type1.csv", "data/synthetic_data_type2.csv", "data/synthetic_data_type3.csv"]
synthetic_data_list = [pd.read_csv(file) for file in synthetic_data_files]

# 평가 결과 저장용 딕셔너리
evaluation_results = {}

# 각 합성 데이터셋 평가
for i, synthetic_data in enumerate(synthetic_data_list):
    print(f"\n##### Evaluating synthetic data: synthetic_data_type{i+1}.csv")
    #synthetic_data = synthetic_data[[col for col in original_data.columns if col != "HNDE_BANK_RPTV_CODE"]]

    # 품질 보고서 생성
    score = CategoricalCAP.compute(
        real_data=original_data.head(6200),
        synthetic_data=synthetic_data.head(6200),
        key_fields=['HNDE_BANK_RPTV_CODE'],
        sensitive_fields=['OPENBANK_RPTV_CODE','FND_TPCD']
    )

    print(score)


