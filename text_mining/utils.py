
# 텍스트 밀도 기반 컬럼 필터링
# text_density_threshold: 컬럼 내 텍스트 데이터의 비율
# unique_text_threshold: 컬럼 내 고유 텍스트 데이터의 최소 개수
def filter_text_columns(df_data, excep_col_names, text_density_threshold=0.5, unique_text_threshold=5):
    df = df_data.drop(columns=excep_col_names)
    print(df)

    # 컬럼별 텍스트 밀도 계산
    text_density = {}
    for col in df.columns:
        # 전체 값 중 비결측값
        non_null_count = df[col].notnull().sum()
        # 텍스트 데이터 개수
        text_count = df[col].apply(lambda x: isinstance(x, str)).sum()
        # 고유 텍스트 개수
        unique_text_count = df[col].apply(lambda x: str(x) if isinstance(x, str) else None).nunique()

        if non_null_count > 0:
            text_density[col] = {
                'text_ratio': text_count / non_null_count,
                'unique_text_count': unique_text_count,
            }

    # 기준에 따라 필터링된 컬럼 선택
    filtered_columns = [
        col for col, stats in text_density.items()
        if stats['text_ratio'] >= text_density_threshold and stats['unique_text_count'] >= unique_text_threshold
    ]
    filtered_columns.extend(excep_col_names)

    return df_data[filtered_columns]
