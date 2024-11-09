import pandas as pd


# 컬럼별 공통된 유니크 값
def get_common_unique_values(dfs, columns):
    common_values = {}
    for col in columns:
        unique_sets = [set(df[col].unique()) for df in dfs]
        common_values[col] = set.intersection(*unique_sets)
        print('#### commin unique')
        print(f'col: {col}, value: [{common_values[col]}]')
    return common_values

# 컬럼별 공통된 유니크 값을 고려한 샘플링
def filter_with_common_values(data, common_values):
    filtered_data = data.copy()
    for col, values in common_values.items():
        filtered_data = filtered_data[filtered_data[col].isin(values)]
    return filtered_data.reset_index(drop=True)


# 대상 데이터셋
file_paths = {
    '100': './datasets/DATOP_HF_TRANS_100.csv',
    '102': './datasets/DATOP_HF_TRANS_102.csv',
    '104': './datasets/DATOP_HF_TRANS_104.csv'
}
target_columns = ['BASE_YM', 'OPENBANK_RPTV_CODE', 'FND_TPCD']

# 공통 유니크 값 추출
dfs = [pd.read_csv(path) for path in file_paths.values()]
common_values = get_common_unique_values(dfs, target_columns)

output_dir = './datasets/'
filtered_data_dict = {}

for key, path in file_paths.items():
    data = pd.read_csv(path)
    filtered_data = filter_with_common_values(data, common_values)

    output_path = f"{output_dir}DATOP_HF_TRANS_{key}_iid.csv"
    filtered_data.to_csv(output_path, index=False)
    filtered_data_dict[key] = filtered_data
    print(f"Filtered data saved for file {key} at {output_path}")


for file_id, sampled_data in filtered_data_dict.items():
    print(f"\nSampled data for file {file_id}:")
    for col in target_columns:
        print(f"{col} unique values: {sorted(sampled_data[col].unique())}")
