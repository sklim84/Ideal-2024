# 필요한 라이브러리 임포트
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import pandas as pd
import os

# 데이터셋 경로 설정
file_paths = ['DATOP_HF_TRANS_AHN.csv',
              'DATOP_HF_TRANS_AIG.csv',
              'DATOP_HF_TRANS_BVI.csv']

# 각 파일에서 100개씩 샘플링하여 데이터프레임으로 읽기
datasets = [pd.read_csv(os.path.join('./datasets', file)).sample(n=100, random_state=42) for file in file_paths]

# 유형1: 샘플링한 데이터를 하나의 데이터프레임으로 통합
combined_data_1 = pd.concat(datasets)
combined_data_1.to_csv('original_data.csv', index=False)


# 메타데이터 생성
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(combined_data_1)
metadata.update_column(
    column_name='BASE_YM',
    sdtype='datetime',
    datetime_format='%Y%m'
)

metadata.update_column(
    column_name='TRAN_AMT',
    sdtype='numerical'
)

metadata.update_column(
    column_name='HNDE_BANK_RPTV_CODE',
    sdtype='categorical'
)

metadata.update_column(
    column_name='OPENBANK_RPTV_CODE',
    sdtype='categorical'
)

metadata.update_column(
    column_name='FND_TPCD',
    sdtype='categorical'
)

# CTGAN 모델 초기화 및 유형1 데이터 학습 (metadata 필요)
ctgan_1 = CTGANSynthesizer(metadata=metadata)
ctgan_1.fit(combined_data_1)

# 유형1: 합성 데이터 300개 생성
synthetic_data_1 = ctgan_1.sample(300)

# 합성 데이터를 파일로 저장 (유형1)
synthetic_data_1.to_csv('synthetic_data_type1.csv', index=False)

# 유형2: 각 데이터셋에서 100개의 합성 데이터를 생성하고 결합
synthetic_data_2_list = []
for dataset in datasets:
    # 개별 데이터셋에 대한 CTGAN 모델 학습
    ctgan_2 = CTGANSynthesizer(metadata=metadata)
    ctgan_2.fit(dataset)

    # 각 데이터셋에서 100개의 합성 데이터를 생성
    synthetic_data_2 = ctgan_2.sample(100)
    synthetic_data_2_list.append(synthetic_data_2)

# 유형2: 각 합성 데이터를 결합
combined_synthetic_data_2 = pd.concat(synthetic_data_2_list)

# 합성 데이터를 파일로 저장 (유형2)
combined_synthetic_data_2.to_csv('synthetic_data_type2.csv', index=False)


