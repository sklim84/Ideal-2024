import pandas as pd

data_file = '../DATOP_PG_TRANS.csv'

df = pd.read_csv(data_file)

# ZWNBSP 제거 (UTF-8 인코딩에서 \ufeff로 표시)
df.columns = df.columns.str.replace('\ufeff', '')  # 헤더에서 ZWNBSP 제거
df = df.applymap(lambda x: x.replace('\ufeff', '') if isinstance(x, str) else x)  # 데이터에서 ZWNBSP 제거

df.to_csv(data_file, index=False)
