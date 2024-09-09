import pandas as pd

# 데이터 로드
file_path = '../DATOP_HF_TRANS.csv'
data = pd.read_csv(file_path)


# 고유값을 100부터 순차적으로 할당하는 함수
def replace_with_unique_sequential_ids(data, column, start_value=100):
    # 고유값에 대해 100부터 시작하는 순차적인 매핑 생성
    unique_values = data[column].unique()
    mapping = {old_value: new_value for new_value, old_value in enumerate(unique_values, start=start_value)}

    # 컬럼에 매핑 적용
    data[column] = data[column].map(mapping)

    return data, mapping


# HNDE_BANK_RPTV_CODE 컬럼 치환
data, hnde_mapping = replace_with_unique_sequential_ids(data, 'HNDE_BANK_RPTV_CODE', start_value=100)

# OPENBANK_RPTV_CODE 컬럼 치환
data, openbank_mapping = replace_with_unique_sequential_ids(data, 'OPENBANK_RPTV_CODE',
                                                            start_value=100 + len(hnde_mapping))

# 결과 확인
print(data.head())
print("HNDE_BANK_RPTV_CODE Mapping:", hnde_mapping)
print("OPENBANK_RPTV_CODE Mapping:", openbank_mapping)

# 데이터 저장
data.to_csv('../DATOP_HF_TRANS_ENC_CODE.csv', index=False)
data[:1000].to_csv('../DATOP_HF_TRANS_ENC_CODE_1000.csv', index=False)