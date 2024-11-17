import copy
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import syft as sy
import torch
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata

# from ctgan import CTGAN
from our_ctgan import CTGAN
from our_data_transformer import DataTransformer
from utils import set_seed

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def initialize_device():
    # FIXME torch 버전 등 이슈로 cpu만 인식 중
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(file_path, num_samples=1000):
    data = pd.read_csv(file_path)[:num_samples]
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
    return data

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

def compute_gradients(model, data_remote, discrete_columns):
    """
    클라이언트에서 데이터를 사용해 모델 학습 후 그라디언트를 계산
    """
    # 모델 복사 (학습 전 파라미터를 유지)
    model_copy = copy.deepcopy(model)
    data_columns = ['BASE_YM', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD', 'TRAN_AMT']
    
    # 데이터가 syft.Tensor 객체일 경우에만 .get() 호출
    if isinstance(data_remote, sy.Tensor):
        data_remote = data_remote.get()  # 데이터를 원격에서 가져옵니다
        data_df = pd.DataFrame(data_remote, columns=data_columns)  # 열 이름을 명시적으로 지정
    else:
        # 이미 numpy.ndarray 형태로 데이터가 로드된 경우
        data_df = pd.DataFrame(data_remote, columns=data_columns)  # 열 이름을 명시적으로 지정

    # 데이터프레임의 열 이름 출력
    print("Data columns:", data_df.columns)

    # 각 열 이름에 공백이 있을 경우를 대비해 strip() 적용
    discrete_columns = [col.strip() for col in discrete_columns]

    # 모델 복사본의 _generator가 정상적으로 초기화되었는지 확인
    if model_copy._generator is None:
        print("Initializing _generator explicitly")
        # discrete_columns의 각 열 이름이 data_df.columns에 존재하는지 확인
        missing_columns = [col for col in discrete_columns if col not in data_df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns in data: {missing_columns}")
        discrete_column_indices = [data_df.columns.get_loc(col) for col in discrete_columns]
        model_copy.fit(data_df, discrete_columns=discrete_columns)

    # 데이터 학습 전에 열 이름을 인덱스로 변환
    discrete_column_indices = [data_df.columns.get_loc(col) for col in discrete_columns]

    # 모델 학습
    model_copy.fit(data_df, discrete_columns=discrete_columns)

    # 이제 _generator가 초기화되었을 것이므로, 파라미터 클론
    initial_params = {name: param.clone().detach() for name, param in model_copy._generator.named_parameters()}

    # 학습 후의 파라미터 저장
    updated_params = {name: param.clone().detach() for name, param in model_copy._generator.named_parameters()}

    # 그라디언트 계산 (학습 후 - 학습 전 파라미터)
    gradients = {name: updated_params[name] - initial_params[name] for name in initial_params.keys()}

    return gradients

def merge_gradients(gradients_list, model):
    num_clients = len(gradients_list)
    merged_gradients = {}

    # 각 그라디언트 항목을 합산하여 평균 계산
    for name in gradients_list[0]:
        merged_gradients[name] = sum(gradients[name] for gradients in gradients_list) / num_clients

    # 모델 파라미터 업데이트 (SGD 방식)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in merged_gradients:
                param.data -= merged_gradients[name]  # 파라미터 업데이트

    # 업데이트된 모델 반환
    return model


if __name__ == "__main__":
    set_seed(2024)

    device = initialize_device()

    num_samples_org = 100  # each / total= x3
    num_samples_syn = 300  # total

    bank_codes = [100, 102, 104]
    datasets = [
        load_data(f'./datasets/DATOP_HF_TRANS_{bank_codes[0]}_iid.csv', num_samples=num_samples_org),
        load_data(f'./datasets/DATOP_HF_TRANS_{bank_codes[1]}_iid.csv', num_samples=num_samples_org),
        load_data(f'./datasets/DATOP_HF_TRANS_{bank_codes[2]}_iid.csv', num_samples=num_samples_org)
    ]

    data_total = pd.concat(datasets, axis=0).reset_index(drop=True)

    total_columns = data_total.columns
    discrete_columns = ['BASE_YM', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD']
    print(data_total)

    bank_groups = [(bank_codes[0], datasets[0]), (bank_codes[1], datasets[1]), (bank_codes[2], datasets[2])]

    # 클라이언트, 데이터 포인터 생성
    clients, data_ptrs = create_clients(bank_groups)

    # 모델 초기화
    initial_model = CTGAN(
        embedding_dim=16,
        generator_dim=(16, 16),
        discriminator_dim=(16, 16),
        batch_size=500,
        epochs=10,
        pac=10
    )

    # 각 클라이언트에서 그라디언트 계산
    gradients_list = []
    for bank_code, client in clients.items():
        print(f'##### bank code:{bank_code} START TRAINING #####')
        data_ptr = data_ptrs[bank_code]
        data_remote = data_ptr.get()
        data_remote = data_remote.child

        gradients = compute_gradients(initial_model, data_remote, discrete_columns)
        gradients_list.append(gradients)

    for i, data in enumerate(gradients_list):
        print(f"Client {i} gradients:")
        for name, grad in data.items():
            print(f"{name}: {grad.shape}")

    # 중앙 서버에서 그라디언트 병합 및 모델 업데이트
    federated_model = merge_gradients(gradients_list, initial_model)

    # 합성 데이터 생성
    synthetic_data = federated_model.sample(num_samples_syn)
    synthetic_data['TRAN_AMT'] = synthetic_data['TRAN_AMT'].abs()
    print("Synthetic Data Generated by the Federated Model:")
    print(synthetic_data)
    synthetic_data.to_csv('data/synthetic_data_type2.csv', index=False)

