import numpy as np
import pandas as pd
import syft as sy
from ctgan import CTGAN
import torch
import copy
from collections import OrderedDict
import glob
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(file_path, num_samples=1000):
    data = pd.read_csv(file_path)[:num_samples]
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
    return data


def train_ctgan(data, emb_dim=32, gen_dim=64, dis_dim=64, epoch=10, pac=10):
    print("Data content (first 5 rows):")
    print(data[:5])

    if hasattr(data, 'child'):
        data = data.child

    if isinstance(data, pd.DataFrame):
        data_list = data.values
    elif isinstance(data, np.ndarray):
        data_list = pd.DataFrame(data.tolist(), columns=['BASE_YM', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD', 'TRAN_AMT'])
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")

    if len(data_list) == 0 or len(data_list[0]) != 5:
        raise ValueError(f"Data does not have the expected shape: {data_list[:5]}")

    columns = ['BASE_YM', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD', 'TRAN_AMT']
    data_df = pd.DataFrame(data_list, columns=columns)

    model = CTGAN(
        embedding_dim=emb_dim,
        generator_dim=(gen_dim, gen_dim),
        discriminator_dim=(dis_dim, dis_dim),
        epochs=epoch,
        pac=pac
    )
    print(model)

    # model.fit(data_df, discrete_columns=['HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD'])
    model.fit(data_df, discrete_columns=['HNDE_BANK_RPTV_CODE'])
    return model


if __name__ == "__main__":
    device = initialize_device()

    # 모든 CSV 파일 경로를 불러오기
    # csv_files = glob.glob('./datasets/*.csv')
    csv_files = [
        './datasets/DATOP_HF_TRANS_100.csv',
        './datasets/DATOP_HF_TRANS_101.csv',
        './datasets/DATOP_HF_TRANS_102.csv'
    ]

    all_synthetic_data = pd.DataFrame()

    for file_path in csv_files:
        print(f"Processing file: {file_path}")
        data = load_data(file_path, num_samples=1000)

        print('total data samples: ')

        model = train_ctgan(data)

        if model:
            synthetic_data = model.sample(100)
            synthetic_data['TRAN_AMT'] = synthetic_data['TRAN_AMT'].abs()
            all_synthetic_data = pd.concat([all_synthetic_data, synthetic_data], ignore_index=True)
        else:
            print(f"Failed to train model for {file_path}")

    # 합성 데이터 전체를 하나의 CSV 파일로 저장
    all_synthetic_data.to_csv('data/synthetic_data_type3.csv', index=False)
    print("Combined Synthetic Data Generated:")
    print(all_synthetic_data)
