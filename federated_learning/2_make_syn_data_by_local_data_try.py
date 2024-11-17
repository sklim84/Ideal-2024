import numpy as np
import pandas as pd
import syft as sy
from ctgan import CTGAN
import torch
import copy
from collections import OrderedDict
import glob
import warnings
from utils import set_seed

warnings.filterwarnings("ignore", category=FutureWarning)


def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(file_path, num_samples=1000):
    data = pd.read_csv(file_path)[:num_samples]
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
    return data


def train_ctgan(data, total_columns, discrete_columns, emb_dim=16, gen_dim=16, dis_dim=16, batch_size=500, epoch=10, pac=10):
    print("Data content (first 5 rows):")
    print(data[:5])

    if hasattr(data, 'child'):
        data = data.child

    if isinstance(data, pd.DataFrame):
        data_list = data.values
    elif isinstance(data, np.ndarray):
        data_list = pd.DataFrame(data.tolist(), columns=total_columns)
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")

    if len(data_list) == 0 or len(data_list[0]) != 5:
        raise ValueError(f"Data does not have the expected shape: {data_list[:5]}")

    # columns = ['BASE_YM', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD', 'TRAN_AMT']
    data_df = pd.DataFrame(data_list, columns=total_columns)

    model = CTGAN(
        embedding_dim=emb_dim,
        generator_dim=(gen_dim, gen_dim),
        discriminator_dim=(dis_dim, dis_dim),
        batch_size=batch_size,
        epochs=epoch,
        pac=pac
    )
    print(model)

    model.fit(data_df, discrete_columns=discrete_columns)
    # model.fit(data_df, discrete_columns=['HNDE_BANK_RPTV_CODE'])
    return model


if __name__ == "__main__":
    set_seed(2024)

    device = initialize_device()

    num_samples_org = 100  # each / total= x3
    num_samples_syn = 100  # each / total= x3

    bank_codes = [100, 102, 104]
    csv_files = [
        f'./datasets/DATOP_HF_TRANS_{bank_codes[0]}_iid.csv',
        f'./datasets/DATOP_HF_TRANS_{bank_codes[1]}_iid.csv',
        f'./datasets/DATOP_HF_TRANS_{bank_codes[2]}_iid.csv'
    ]

    all_synthetic_data = pd.DataFrame()

    for file_path in csv_files:
        print(f"Processing file: {file_path}")
        data = load_data(file_path, num_samples=num_samples_org)

        print('total data samples: ')

        total_columns = data.columns
        discrete_columns = ['BASE_YM', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD']

        # model = train_ctgan(data)
        model = train_ctgan(data=data,
                            total_columns=total_columns,
                            discrete_columns=discrete_columns,
                            emb_dim=16,
                            gen_dim=16, dis_dim=16,
                            batch_size=500,
                            epoch=10, pac=10)

        if model:
            synthetic_data = model.sample(num_samples_syn)
            synthetic_data['TRAN_AMT'] = synthetic_data['TRAN_AMT'].abs()
            all_synthetic_data = pd.concat([all_synthetic_data, synthetic_data], ignore_index=True)
        else:
            print(f"Failed to train model for {file_path}")

    # 합성 데이터 전체를 하나의 CSV 파일로 저장
    all_synthetic_data.to_csv('data/synthetic_data_type3.csv', index=False)
    print("Combined Synthetic Data Generated:")
    print(all_synthetic_data)
