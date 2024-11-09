import pandas as pd

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import warnings

warnings.filterwarnings("ignore")

# 원본 데이터 로드
original_data = pd.read_csv("../synthetic_data/org_datasets/DATOP_HF_TRANS_ENC_CODE.csv")
# original_data = original_data[[col for col in original_data.columns if col != "HNDE_BANK_RPTV_CODE"]]

# 세 개의 합성 데이터 로드
synthetic_data_files = ["data/synthetic_data_type1.csv", "data/synthetic_data_type2.csv", "data/synthetic_data_type3.csv"]
synthetic_data_list = [pd.read_csv(file) for file in synthetic_data_files]

# 평가 결과 저장용 딕셔너리
evaluation_results = {}

# 메타데이터 설정
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(original_data)

# 열 유형을 지정 (예시)
metadata.update_column("BASE_YM", sdtype="datetime", datetime_format="%Y%m")
metadata.update_column("TRAN_AMT", sdtype="numerical")
metadata.update_column("HNDE_BANK_RPTV_CODE", sdtype="numerical")
metadata.update_column("OPENBANK_RPTV_CODE", sdtype="numerical")
metadata.update_column("FND_TPCD", sdtype="numerical")

# 각 합성 데이터셋 평가
for i, synthetic_data in enumerate(synthetic_data_list):
    print(f"\n##### Evaluating synthetic data: synthetic_data_type{i+1}.csv")
    # synthetic_data = synthetic_data[[col for col in original_data.columns if col != "HNDE_BANK_RPTV_CODE"]]

    #if(i == 2):
    #    original_data = pd.read_csv("../datasets/DATOP_HF_TRANS_100.csv")

    # 품질 보고서 생성
    quality_report = evaluate_quality(original_data, synthetic_data, metadata)

    # 평가 점수 계산 (품질 점수와 유효성 점수)
    #quality_score = evaluate(original_data, synthetic_data)
    validity_score = quality_report.get_score()

    # 세부 품질 평가 결과
    column_shapes = quality_report.get_details("Column Shapes")
    column_pair_trends = quality_report.get_details("Column Pair Trends")

    # 결과 저장
    evaluation_results[f"synthetic_data_type{i+1}"] = {
        "Validity Score": validity_score,
        "Column Shapes": column_shapes,
        "Column Pair Trends": column_pair_trends,
    }

# 최종 평가 결과 출력
for key, result in evaluation_results.items():
    print(f"\n##### Results for {key}:")
    print(f"Validity Score: {result['Validity Score']}")
    print("Column Shapes:\n", result["Column Shapes"])
#    print("Column Pair Trends:\n", result["Column Pair Trends"])

