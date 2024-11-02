import pandas as pd

# Load the dataset
#file_path = './datasets/DATOP_HF_TRANS.csv'
file_path = '../synthetic_data/org_datasets/DATOP_HF_TRANS_ENC_CODE.csv'
data = pd.read_csv(file_path)

# Split the dataframe by 'HNDE_BANK_RPTV_CODE'
grouped_data = data.groupby('HNDE_BANK_RPTV_CODE')

# Save each group as a separate CSV file
for name, group in grouped_data:
    output_file = f'./datasets/DATOP_HF_TRANS_{name}.csv'
    group.to_csv(output_file, index=False)

output_file_names = [f'DATOP_HF_TRANS_{name}.csv' for name in grouped_data.groups.keys()]
print(output_file_names)
