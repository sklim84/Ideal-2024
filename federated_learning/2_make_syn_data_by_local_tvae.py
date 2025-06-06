import warnings

import numpy as np
import pandas as pd
import torch
from ctgan.synthesizers.tvae import TVAE

from utils import set_seed, evaluate_syn_data
from config import get_config
import os

warnings.filterwarnings("ignore")


def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(file_path, num_samples=1000):
    data = pd.read_csv(file_path)[:num_samples]
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
    return data


def train_tvae(data, total_columns, discrete_columns, emb_dim=16, gen_dim=16, dis_dim=16, batch_size=500, epoch=10):
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

    model = TVAE(
        embedding_dim=emb_dim,
        compress_dims=(gen_dim, gen_dim),
        decompress_dims=(dis_dim, dis_dim),
        batch_size=batch_size,
        epochs=epoch
    )

    print(model)

    model.fit(data_df, discrete_columns=discrete_columns)
    # model.fit(data_df, discrete_columns=['HNDE_BANK_RPTV_CODE'])
    return model


if __name__ == "__main__":
    args = get_config()
    print(args)

    set_seed(args.seed)

    device = initialize_device()

    num_samples_org = int(args.num_samples_org / 3)
    num_samples_syn = int(args.num_samples_syn / 3)

    bank_codes = [100, 102, 104]
    csv_files = [
        f'./datasets/DATOP_HF_TRANS_{bank_codes[0]}_iid.csv',
        f'./datasets/DATOP_HF_TRANS_{bank_codes[1]}_iid.csv',
        f'./datasets/DATOP_HF_TRANS_{bank_codes[2]}_iid.csv'
    ]

    all_synthetic_data = pd.DataFrame()
    
    for idx, file_path in enumerate(csv_files):
        print(f"Processing file: {file_path}")
        data = load_data(file_path, num_samples=num_samples_org)

        print('total data samples: ')

        total_columns = data.columns
        discrete_columns = ['BASE_YM', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD']

        # model = train_ctgan(data)
        model = train_tvae(data=data,
                           total_columns=total_columns,
                           discrete_columns=discrete_columns,
                           emb_dim=args.emb_dim,
                           gen_dim=args.gen_dim, dis_dim=args.dis_dim,
                           batch_size=args.batch_size,
                           epoch=args.epoch)

        if model:
            synthetic_data = model.sample(num_samples_syn)
            synthetic_data['TRAN_AMT'] = synthetic_data['TRAN_AMT'].abs()
        else:
            print(f"Failed to train model for {file_path}")

        syn_data_path = f'./datasets_syn/syn_type_lo_tave_to_{num_samples_org*3}_to_{num_samples_syn}_{bank_codes[idx]}.csv'
        synthetic_data.to_csv(syn_data_path, index=False)

        # evaluation
        df_results = evaluate_syn_data('./results/eval_results.csv',
                                   './datasets/DATOP_HF_TRANS_100_102_104_iid.csv',
                                   syn_data_path,
                                   model_name='tvae',
                                   method='local',
                                   num_org=num_samples_org*3,
                                   num_syn=num_samples_syn)
        print(df_results)
