# 필요한 라이브러리 임포트
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import get_column_plot
import pandas as pd
# 필요한 라이브러리 임포트
import os
import plotly.io as pio
from scipy.stats import chi2_contingency

# 합성 데이터 파일 경로
synthetic_data_file_type1 = 'synthetic_data_type1.csv'
synthetic_data_file_type2 = 'synthetic_data_type2.csv'

# 원본 데이터 파일 경로
original_data_file = 'original_data.csv'  # 원본 데이터가 저장된 경로 (필요에 따라 수정)

# 합성 데이터 및 원본 데이터 불러오기
synthetic_data_1 = pd.read_csv(synthetic_data_file_type1)
combined_synthetic_data_2 = pd.read_csv(synthetic_data_file_type2)
combined_data_1 = pd.read_csv(original_data_file)

# 품질 분석 함수 정의: 원본 데이터와 합성 데이터의 평균 및 표준편차 비교
def analyze_quality(original_data, synthetic_data):
    numerical_columns = original_data.select_dtypes(include=['number']).columns
    analysis = pd.DataFrame({
        "Original_Mean": original_data[numerical_columns].mean(),
        "Synthetic_Mean": synthetic_data[numerical_columns].mean(),
        "Original_Std": original_data[numerical_columns].std(),
        "Synthetic_Std": synthetic_data[numerical_columns].std(),
    })
    return analysis

# 유형1 품질 분석
quality_analysis_1 = analyze_quality(combined_data_1, synthetic_data_1)

# 유형2 품질 분석 (통합된 원본 데이터와 합성 데이터 비교)
quality_analysis_2 = analyze_quality(combined_data_1, combined_synthetic_data_2)

# 결과 출력
print("Quality Analysis for 유형1:\n", quality_analysis_1)
print("Quality Analysis for 유형2:\n", quality_analysis_2)

# 범주형 데이터의 유사성을 측정하는 함수 (Chi-Squared Test)
def calculate_chi2(original_data, synthetic_data, column_name):
    # 실제 데이터와 합성 데이터의 분포를 crosstab으로 만듭니다.
    contingency_table = pd.crosstab(original_data[column_name], synthetic_data[column_name])

    # 카이제곱 검정 수행
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return chi2, p_value

# 'OPENBANK_RPTV_CODE' 열에 대한 카이제곱 검정 수행
chi2_result1, p_value1 = calculate_chi2(combined_data_1, synthetic_data_1, 'OPENBANK_RPTV_CODE')

# 'OPENBANK_RPTV_CODE' 열에 대한 카이제곱 검정 수행
chi2_result2, p_value2 = calculate_chi2(combined_data_1, combined_synthetic_data_2, 'OPENBANK_RPTV_CODE')

# 결과 출력
print(f"1 Chi-squared test result for 'OPENBANK_RPTV_CODE': Chi2 = {chi2_result1}, p-value = {p_value1}")
print(f"2 Chi-squared test result for 'OPENBANK_RPTV_CODE': Chi2 = {chi2_result2}, p-value = {p_value2}")


# 시각화 함수: 특정 열에 대해 원본 데이터와 합성 데이터를 비교하는 그래프 생성 및 저장
# 시각화 함수: 특정 열에 대해 원본 데이터와 합성 데이터를 비교하는 그래프 생성 및 저장
def visualize_column(original_data, synthetic_data, metadata, column_name, output_dir, file_name):
    fig = get_column_plot(
        real_data=original_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        column_name=column_name
    )

    # y축 최대값을 설정하는 옵션
    if column_name == 'OPENBANK_RPTV_CODE':
        fig.update_layout(
            yaxis_range=[0, 0.06]  # y축 범위 설정 (0부터 y_max까지)
        )

    # # y축 로그 스케일 설정, 하지만 0에 가까운 값에 대해 조정
    # fig.update_layout(
    #     yaxis_type="log",  # 로그 스케일 적용
    #     title=f'Log-scaled Frequency for {column_name}',
    #     yaxis_title='Log Frequency'
    # )

    # 파일 저장 경로
    file_path = os.path.join(output_dir, f'{file_name}_{column_name}_comparison.png')

    # plotly fig를 PNG로 저장
    fig.write_image(file_path)
    print(f"Visualization for {column_name} saved as {file_path}")

# 메타데이터 생성 (합성 데이터에서 필요할 경우)
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
# 각 열에 대해 시각화 수행
columns_to_visualize = ['TRAN_AMT', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE']  # 시각화할 열 목록

# 시각화 결과 저장 경로
output_directory = './visualizations'
os.makedirs(output_directory, exist_ok=True)

# 각 열에 대해 시각화 수행 및 PNG로 저장
for column in columns_to_visualize:
    print(f"Visualizing and saving {column}")
    visualize_column(combined_data_1, synthetic_data_1, metadata, column, output_directory, 'syn1')
    visualize_column(combined_data_1, combined_synthetic_data_2, metadata, column, output_directory, 'syn2')
