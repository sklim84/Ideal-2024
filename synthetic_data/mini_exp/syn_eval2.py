import pandas as pd
import plotly.figure_factory as ff
import os

# 합성 데이터 및 원본 데이터 불러오기
synthetic_data_1 = pd.read_csv('synthetic_data_type1.csv')
synthetic_data_2 = pd.read_csv('synthetic_data_type2.csv')
original_data = pd.read_csv('original_data.csv')

# 시각화할 열 선택 (예: 'TRAN_AMT')
column_to_visualize = 'TRAN_AMT'

# 각 데이터셋에 대해 리스트로 데이터 준비
data_original = original_data[column_to_visualize].dropna().tolist()
data_synthetic_1 = synthetic_data_1[column_to_visualize].dropna().tolist()
data_synthetic_2 = synthetic_data_2[column_to_visualize].dropna().tolist()

# 연속 분포 (KDE) 플롯 생성
fig = ff.create_distplot(
    [data_original, data_synthetic_1, data_synthetic_2],
    group_labels=['Original Data', 'Synthetic Data Type 1', 'Synthetic Data Type 2'],
    show_hist=False,  # 히스토그램 제거
    show_rug=False,   # 밀도 플롯 아래에 작은 선 제거
)

# 레이아웃 업데이트
fig.update_layout(
    title=f'Comparison of {column_to_visualize}',
    xaxis_title=column_to_visualize,
    yaxis_title='Density',
    legend_title="Dataset"
)

# PNG 파일로 저장할 경로 설정
output_directory = './visualizations'
os.makedirs(output_directory, exist_ok=True)  # 폴더가 없으면 생성
file_path = os.path.join(output_directory, f'comparison_{column_to_visualize}_kde.png')

# 그래프를 PNG로 저장
fig.write_image(file_path)
print(f"Visualization saved as {file_path}")
