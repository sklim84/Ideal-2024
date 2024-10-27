import numpy as np
import pandas as pd
import syft as sy
from ctgan import CTGAN
import torch
import copy
from collections import OrderedDict
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def initialize_device():
    # FIXME torch 버전 등 이슈로 cpu만 인식 중
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(file_path, num_samples=1000):
    data = pd.read_csv(file_path)[:num_samples]
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int32)
    return data


def create_clients(data):
    clients = {}
    data_ptrs = {}

    # FIXME 취급은행코드 기준 분할 시 분할된 데이터에서 data unique 값들이 달라져 CTGAN shape이 맞지 않는 문제 발생
    # FIXME 취급은행별로 데이터를 분할하지 않고, data unique가 충족되도록 분할하는 것은 어떨지
    # 취급은행코드 기준 데이터 분할
    bank_groups = data.groupby('HNDE_BANK_RPTV_CODE')

    for bank_code, bank_data in bank_groups:
        if len(bank_data) < 2:
            print(f"Skipping bank code {bank_code}: insufficient samples ({len(bank_data)} samples).")
            continue

        vm = sy.VirtualMachine(name=f"client_{bank_code}")
        client = vm.get_root_client()
        clients[bank_code] = client

        data_array = bank_data.values.astype(np.int32)
        tensor = sy.Tensor(data_array)
        data_ptrs[bank_code] = tensor.send(client)

    return clients, data_ptrs


def train_ctgan(data, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256), pac=10):
    print("Data content (first 5 rows):")
    print(data[:5])

    if hasattr(data, 'child'):
        data = data.child

    if isinstance(data, np.ndarray):
        data_list = data.tolist()
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")

    if len(data_list) == 0 or len(data_list[0]) != 5:
        raise ValueError(f"Data does not have the expected shape: {data_list[:5]}")

    columns = ['BASE_YM', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD', 'TRAN_AMT']
    data_df = pd.DataFrame(data_list, columns=columns)

    model = CTGAN(
        embedding_dim=embedding_dim,
        generator_dim=generator_dim,
        discriminator_dim=discriminator_dim,
        pac=pac,
        epochs=10
    )
    print(model)

    # FIXME discrete_columns 설정 (OPENBANK_RPTV_CODE,FND_TPCD 추가)
    model.fit(data_df, discrete_columns=['HNDE_BANK_RPTV_CODE'])
    return model


def pad_tensor(tensor, target_shape):
    current_shape = tensor.shape
    if current_shape == target_shape:
        return tensor

    padded_tensor = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, min(current, target)) for current, target in zip(current_shape, target_shape))
    padded_tensor[slices] = tensor[slices]

    return padded_tensor


def merge_models(models, model_details):
    if not models:
        return None

    # FIXME models[0]을 기준으로 하는 문제 : HNDE_BANK_RPTV_CODE=100만 생성됨
    merged_model = copy.deepcopy(models[0])
    merged_state_dict = OrderedDict()

    num_models = len(models)

    for model in models:
        model_state = model._generator.state_dict()

        for key in model_state:

            print(f'load model: {model}, key: {key}')

            target_shape = merged_state_dict[key].shape if key in merged_state_dict else model_state[key].shape
            # FIXME 모델 통합 시 차원 불일치 이슈로 차원을 맞추는 패딩 로직 추가
            padded_tensor = pad_tensor(model_state[key], target_shape)

            if key not in merged_state_dict:
                merged_state_dict[key] = padded_tensor.clone()
            else:
                if merged_state_dict[key].shape != padded_tensor.shape:
                    print(
                        f"Error merging key {key}: Parameter size mismatch. {merged_state_dict[key].shape} vs {padded_tensor.shape}")
                    for bank_code, mdl in model_details.items():
                        if mdl is model:
                            print(f"Error originating from Bank Code: {bank_code}")
                            print(f"Model source key: {key}, Model: {model}")
                            break
                    return None
                merged_state_dict[key] = merged_state_dict[key].float() + padded_tensor.float()

    # 파라미터 평균값 계산(FedAVG)
    for key in merged_state_dict:
        merged_state_dict[key] /= num_models
        merged_state_dict[key] = merged_state_dict[key].type(model_state[key].dtype)

    merged_model._generator.load_state_dict(merged_state_dict)

    return merged_model


if __name__ == "__main__":
    device = initialize_device()

    # 타행이체거래내역
    file_path = '../synthetic_data/org_datasets/DATOP_HF_TRANS_ENC_CODE.csv'
    data = load_data(file_path)

    # 클라이언트, 데이터 포인터 생성
    clients, data_ptrs = create_clients(data)

    # 각 클라이언트의 모델 학습
    model_ptrs = []
    model_details = {}

    for bank_code, client in clients.items():
        try:
            print(f'##### bank code:{bank_code} START TRAINING #####')
            data_ptr = data_ptrs[bank_code]
            data_remote = data_ptr.get()

            if isinstance(data_remote, sy.Tensor):
                data_remote = data_remote.child

            model = train_ctgan(data_remote)
            model_ptrs.append(model)
            model_details[bank_code] = model

        except Exception as e:
            print(f"Error processing bank code {bank_code}: {e}")

    # Merge models
    federated_model = merge_models(model_ptrs, model_details)

    if federated_model:
        synthetic_data = federated_model.sample(10)
        print("Synthetic Data Generated by the Federated Model:")
        print(synthetic_data)
    else:
        print("No models were successfully trained.")
