import copy
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import syft as sy
import torch

from our_ctgan import CTGAN
from our_data_transformer import DataTransformer
from utils import set_seed, evaluate_syn_data, parse_args

warnings.filterwarnings("ignore")


def print_object_details(obj):
    print("Object details:")
    for attribute, value in obj.__dict__.items():
        print(f"{attribute}: {value}")


def print_generator_out_features(ctgan_model):
    print("Out features in each layer of Generator:")
    for layer_name, layer in ctgan_model._generator.seq.named_children():
        if isinstance(layer, torch.nn.Linear):
            print(f"{layer_name}: Linear layer out_features = {layer.out_features}")
        elif hasattr(layer, 'fc') and isinstance(layer.fc, torch.nn.Linear):
            print(f"{layer_name}: Residual fc layer out_features = {layer.fc.out_features}")


def initialize_device():
    # FIXME torch 버전 등 이슈로 cpu만 인식 중
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(file_path, num_samples=1000):
    data = pd.read_csv(file_path)[:num_samples]
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
    return data


# def create_clients(data):
def create_clients(bank_groups):
    clients = {}
    data_ptrs = {}

    for bank_code, bank_data in bank_groups:
        vm = sy.VirtualMachine(name=f"client_{bank_code}")
        client = vm.get_root_client()
        clients[bank_code] = client

        data_array = bank_data.values.astype(np.int32)
        tensor = sy.Tensor(data_array)
        data_ptrs[bank_code] = tensor.send(client)

    return clients, data_ptrs


def pad_tensor(tensor, target_shape):
    current_shape = tensor.shape
    if current_shape == target_shape:
        return tensor

    padded_tensor = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, min(current, target)) for current, target in zip(current_shape, target_shape))
    padded_tensor[slices] = tensor[slices]

    return padded_tensor


def train_ctgan(data, total_columns, discrete_columns, emb_dim=32, gen_dim=64, dis_dim=64, batch_size=500, epoch=10,
                pac=10):
    print("Data content (first 5 rows):")
    print(data[:5])

    data_list = data.tolist()
    data_df = pd.DataFrame(data_list, columns=total_columns)

    model = CTGAN(
        embedding_dim=emb_dim,
        generator_dim=(gen_dim, gen_dim),
        discriminator_dim=(dis_dim, dis_dim),
        batch_size=batch_size,
        epochs=epoch,
        pac=pac
    )
    # print(model)

    model.fit(data_df, discrete_columns=discrete_columns)
    return model


def initialize_empty_model(model_template):
    empty_model = copy.deepcopy(model_template)
    empty_state_dict = OrderedDict(
        (key, torch.zeros_like(param)) for key, param in model_template._generator.state_dict().items())
    empty_model._generator.load_state_dict(empty_state_dict, strict=False)
    return empty_model


def merge_data_transformer(train_data, discrete_columns):
    data_transformer = DataTransformer()
    data_transformer.fit(train_data, discrete_columns)
    return data_transformer


def merge_models(models, merged_data_transformer):
    # print_object_details(models[0])
    # print_generator_out_features(models[0])
    # print("---------------------------------")
    # print_object_details(models[1])
    # print_generator_out_features(models[1])

    # models[0] 모델 구조 복사, 빈 모델(가중치=0) 생성
    merged_model = initialize_empty_model(models[0])
    merged_model._transformer = merged_data_transformer

    # merged_model = copy.deepcopy(models[0])
    merged_state_dict = OrderedDict()

    num_models = len(models)
    print(num_models)

    for model in models:
        model_state = model._generator.state_dict()

        for key in model_state:
            print(f'##### load model: {model}, key: {key}')

            target_shape = merged_state_dict[key].shape if key in merged_state_dict else model_state[key].shape
            # 모델 차원을 맞추는 패딩 추가
            padded_tensor = pad_tensor(model_state[key], target_shape)

            # 모델 파라미터 merge
            if key not in merged_state_dict:
                print(f'##### not in merged_state_dict: {model}, key: {key}')
                merged_state_dict[key] = padded_tensor.float()
            else:
                merged_state_dict[key] += padded_tensor.float()

    # 파라미터 평균 계산(FedAVG)
    for key in merged_state_dict:
        print(f'##### merge step key: {key}')
        # merged_state_dict[key] = merged_state_dict[key].float() / num_models
        # merged_state_dict[key] = merged_state_dict[key].type(model_state[key].dtype)
        merged_state_dict[key] = (merged_state_dict[key] / num_models).type(model_state[key].dtype)

    # 병합된 파라미터 로드
    merged_model._generator.load_state_dict(merged_state_dict, strict=False)

    return merged_model


if __name__ == "__main__":
    args = parse_args()
    set_seed(2024)

    device = initialize_device()

    #num_samples_org = 100  # each / total= x3
    #num_samples_syn = 300  # total

    num_samples_org = args.num_samples_org
    num_samples_syn = args.num_samples_syn

    # FIX
    bank_codes = [100, 102, 104]
    datasets = [
        load_data(f'./datasets/DATOP_HF_TRANS_{bank_codes[0]}_iid.csv', num_samples=num_samples_org),
        load_data(f'./datasets/DATOP_HF_TRANS_{bank_codes[1]}_iid.csv', num_samples=num_samples_org),
        load_data(f'./datasets/DATOP_HF_TRANS_{bank_codes[2]}_iid.csv', num_samples=num_samples_org)
    ]

    data_total = pd.concat(datasets, axis=0).reset_index(drop=True)
    # data_total.to_csv(f'./datasets/DATOP_HF_TRANS_{bank_codes[0]}_{bank_codes[1]}_{bank_codes[2]}_iid.csv', index=False)

    total_columns = data_total.columns
    discrete_columns = ['BASE_YM', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD']
    print(data_total)

    bank_groups = [(bank_codes[0], datasets[0]), (bank_codes[1], datasets[1]), (bank_codes[2], datasets[2])]

    # 클라이언트, 데이터 포인터 생성
    clients, data_ptrs = create_clients(bank_groups)

    # 각 클라이언트의 모델 학습
    model_ptrs = []
    for bank_code, client in clients.items():
        print(f'##### bank code:{bank_code} START TRAINING #####')
        data_ptr = data_ptrs[bank_code]
        data_remote = data_ptr.get()
        data_remote = data_remote.child
        # TODO add grid search
        model = train_ctgan(data=data_remote,
                            total_columns=data_total.columns,
                            discrete_columns=discrete_columns,
                            emb_dim=16,
                            gen_dim=16, dis_dim=16,
                            epoch=10, pac=10,
                            batch_size=500)
        model_ptrs.append(model)

    # Merge data transformer
    data_transformer = merge_data_transformer(data_total, discrete_columns)

    # Merge models
    federated_model = merge_models(model_ptrs, data_transformer)

    synthetic_data = federated_model.sample(num_samples_syn)
    synthetic_data['TRAN_AMT'] = synthetic_data['TRAN_AMT'].abs()
    print("Synthetic Data Generated by the Federated Model:")
    print(synthetic_data)

    syn_data_path = f'./datasets_syn/syn_type_fl_avg_ctgan_{num_samples_org * 3}_to_{num_samples_syn}.csv'
    synthetic_data.to_csv(syn_data_path, index=False)

    # evaluation
    df_results = evaluate_syn_data('./results/eval_results.csv',
                                   './datasets/DATOP_HF_TRANS_100_102_104_iid.csv',
                                   syn_data_path,
                                   model_name='ctgan',
                                   method='fedavg',
                                   num_org=num_samples_org * 3,
                                   num_syn=num_samples_syn)
    print(df_results)
