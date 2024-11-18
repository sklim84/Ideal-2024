import copy
import warnings

import numpy as np
import pandas as pd
import syft as sy
import torch
from ctgan.data_sampler import DataSampler

from our_ctgan import CTGAN
from our_data_transformer import DataTransformer
from utils import set_seed

warnings.filterwarnings("ignore")


def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(file_path, num_samples=1000):
    data = pd.read_csv(file_path)[:num_samples]
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
    return data

def pad_tensor(tensor, target_shape):
    current_shape = tensor.shape
    if current_shape == target_shape:
        return tensor

    padded_tensor = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, min(current, target)) for current, target in zip(current_shape, target_shape))
    padded_tensor[slices] = tensor
    return padded_tensor

def initialize_generator(model, embedding_dim, generator_dim, data_dim):
    """
    Reinitializes the generator to match the expected input and output dimensions.
    Adds debug information for validation.
    """
    print(f"Initializing generator: embedding_dim={embedding_dim}, generator_dim={generator_dim}, data_dim={data_dim}")
    model._generator = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim + model._data_sampler.dim_cond_vec(), generator_dim[0]),
        torch.nn.ReLU(),
        torch.nn.Linear(generator_dim[0], generator_dim[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(generator_dim[1], data_dim)
    )


def initialize_data_sampler(model, train_data):
    """
    Initializes the _data_sampler for the given model using train_data.
    Adds debug information for validation.
    """
    transformed_data = model._transformer.transform(train_data)
    print("Transformed data for DataSampler initialization:", transformed_data.shape)
    model._data_sampler = DataSampler(
        transformed_data,
        model._transformer.output_info_list,
        log_frequency=True
    )


def merge_data_transformer(train_data, discrete_columns):
    data_transformer = DataTransformer()
    data_transformer.fit(train_data, discrete_columns)
    return data_transformer


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


def compute_gradients(model, data_remote, discrete_columns):
    model_copy = copy.deepcopy(model)

    data_columns = ['BASE_YM', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD', 'TRAN_AMT']
    data_remote = data_remote.child
    data_remote = pd.DataFrame(data_remote, columns=data_columns)

    # Forward pass to calculate loss
    model_copy.fit(data_remote, discrete_columns=discrete_columns)
    fake_data = model_copy.sample(len(data_remote))

    # Convert to tensors with requires_grad=True
    real_data_tensor = torch.tensor(data_remote.values, dtype=torch.float32, requires_grad=True)
    fake_data_tensor = torch.tensor(fake_data.values, dtype=torch.float32)

    # Calculate loss
    loss = torch.nn.MSELoss()(real_data_tensor, fake_data_tensor)

    # Backward pass to calculate gradients
    loss.backward()

    # Collect gradients
    gradients = {name: param.grad.clone().detach() for name, param in model_copy._generator.named_parameters()}

    return gradients

def merge_gradients(gradients_list, model):
    num_clients = len(gradients_list)
    merged_gradients = {}

    # Determine the maximum shape for each gradient
    max_shapes = {
        name: max(grad[name].shape for grad in gradients_list)
        for name in gradients_list[0]
    }

    # Average gradients across clients
    for name in gradients_list[0]:
        padded_gradients = [
            pad_tensor(grad[name], max_shapes[name]) for grad in gradients_list
        ]
        merged_gradients[name] = sum(padded_gradients) / num_clients

    # Update the model parameters using the averaged gradients
    with torch.no_grad():
        for name, param in model._generator.named_parameters():
            if name in merged_gradients:
                param.data -= merged_gradients[name]

    # Ensure the _data_sampler is initialized
    if model._data_sampler is None:
        initialize_data_sampler(model, train_data=data_total)  # Pass the global data_total

    # Ensure the generator matches the expected input and output dimensions
    initialize_generator(model, embedding_dim=16, generator_dim=(16, 16), data_dim=model._transformer.output_dimensions)

    return model




if __name__ == "__main__":
    set_seed(2024)
    device = initialize_device()

    # Load and preprocess data
    num_samples_org = 100
    num_samples_syn = 300

    bank_codes = [100, 102, 104]
    datasets = [
        load_data(f'./datasets/DATOP_HF_TRANS_{bank_codes[0]}_iid.csv', num_samples=num_samples_org),
        load_data(f'./datasets/DATOP_HF_TRANS_{bank_codes[1]}_iid.csv', num_samples=num_samples_org),
        load_data(f'./datasets/DATOP_HF_TRANS_{bank_codes[2]}_iid.csv', num_samples=num_samples_org)
    ]

    data_total = pd.concat(datasets, axis=0).reset_index(drop=True)
    discrete_columns = ['BASE_YM', 'HNDE_BANK_RPTV_CODE', 'OPENBANK_RPTV_CODE', 'FND_TPCD']

    bank_groups = [(bank_codes[0], datasets[0]), (bank_codes[1], datasets[1]), (bank_codes[2], datasets[2])]

    # Create clients and data pointers
    clients, data_ptrs = create_clients(bank_groups)

    # Initialize CTGAN template and empty model
    template_model = CTGAN(
        embedding_dim=16,
        generator_dim=(16, 16),
        discriminator_dim=(16, 16),
        batch_size=500,
        epochs=1,
        pac=10
    )


    def initialize_empty_model(model_template):
        empty_model = copy.deepcopy(model_template)
        if empty_model._generator is None:
            data_dim = 1  # Placeholder
            empty_model._generator = torch.nn.Sequential(
                torch.nn.Linear(model_template._embedding_dim, model_template._generator_dim[0]),
                torch.nn.ReLU()
            )
        return empty_model


    initial_model = initialize_empty_model(template_model)

    # Compute gradients on each client
    gradients_list = []
    for bank_code, client in clients.items():
        data_remote = data_ptrs[bank_code]
        if hasattr(data_remote, "get"):
            data_remote = data_remote.get()
        gradients = compute_gradients(initial_model, data_remote, discrete_columns)
        gradients_list.append(gradients)

    # Merge data transformer
    data_transformer = merge_data_transformer(data_total, discrete_columns)
    initial_model._transformer = data_transformer

    # Merge gradients and update the global model
    federated_model = merge_gradients(gradients_list, initial_model)

    # Transform the data and initialize _data_sampler
    initialize_data_sampler(federated_model, data_total)

    # Generate synthetic data
    synthetic_data = federated_model.sample(num_samples_syn)
    synthetic_data['TRAN_AMT'] = synthetic_data['TRAN_AMT'].abs()
    print("Synthetic Data Generated by the Federated Model:")
    print(synthetic_data)
    synthetic_data.to_csv('data/synthetic_data_fedsgd.csv', index=False)
