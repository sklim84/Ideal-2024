import pandas as pd

data_file = '../DATOP_CASH_CHECK.csv'

df = pd.read_csv(data_file)
df_sorted = df.sort_values(by="BASE_YM", ascending=True)
print(len(df), len(df_sorted))
print(df_sorted.tail(1))

df_sorted.to_csv(data_file, index=False)
