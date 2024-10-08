import torch
import numpy as np
import itertools
from tqdm import tqdm
from src.signal_creation import Samples
from pathlib import Path
from src.system_model import SystemModelParams
from src.utils import device


def create_dataset(system_model_params: SystemModelParams,
                   samples_size: float, tau: int, model_type: str, save_datasets: bool = True,
                   datasets_path: Path = None, true_doa: list = None, phase: str = None):
    """
    Generates a synthetic datasets based on the specified parameters and model type.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams
        samples_size (float): The size of the datasets.
        tau (int): The number of lags for auto-correlation (relevant only for SubspaceNet model).
        model_type (str): The type of the model.
        save_datasets (bool, optional): Specifies whether to save the datasets. Defaults to False.
        datasets_path (Path, optional): The path for saving the datasets. Defaults to None.
        true_doa (list, optional): Predefined angles. Defaults to None.
        phase (str, optional): The phase of the datasets (test or training phase for CNN model). Defaults to None.

    Returns:
    --------
        tuple: A tuple containing the desired datasets comprised of (X-samples, Y-labels).

    """
    generic_dataset = []
    model_dataset = []

    snr_list = list(np.linspace(-5, 20, 26, dtype=int))
    T_list = list(np.linspace(10, 200, 20, dtype=int))
    samples_model = Samples(system_model_params)
    # Generate permutations for CNN model training datasets
    if model_type.startswith("DeepCNN") and phase.startswith("train"):
        doa_permutations = []
        angles_grid = np.linspace(start=-90, stop=90, num=361)
        for comb in itertools.combinations(angles_grid, system_model_params.M):
            doa_permutations.append(list(comb))

    if model_type.startswith("DeepCNN") and phase.startswith("train"):
        for i, doa in tqdm(enumerate(doa_permutations)):
            # Samples model creation
            samples_model.set_doa(doa, system_model_params.M)

            num = system_model_params.M
            snr = snr_list[i % len(snr_list)]
            T = system_model_params.T

            # Observations matrix creation
            X = torch.tensor(samples_model.samples_creation(source_num=num, noise_mean=0, noise_variance=1,
                                                            signal_mean=0, signal_variance=1, snr=snr, snapshot=T)[0],
                             dtype=torch.complex64)
            X_model = create_cov_tensor(X)
            # Ground-truth creation
            Y = torch.zeros_like(torch.tensor(angles_grid))
            for angle in doa:
                Y[list(angles_grid).index(angle)] = 1
            model_dataset.append((X_model, Y))
            generic_dataset.append((X, Y))

    elif model_type.startswith("Trans2DMUSIC"):
        for i in tqdm(range(samples_size)):
            # Samples model creation
            num = system_model_params.M
            snr = system_model_params.snr  # snr_list[i % len(snr_list)]
            T = system_model_params.T

            samples_model.set_doa(true_doa, num)

            # Observations matrix creation
            X = torch.tensor(samples_model.samples_creation(source_num=num, noise_mean=0, noise_variance=1,
                                                            signal_mean=0, signal_variance=1, snr=snr, snapshot=T)[0],
                             dtype=torch.complex64)
            X_model = create_autocorrelation_tensor(X, tau).to(torch.float)
            # Ground-truth creation
            Y1 = torch.tensor(samples_model.doa_t, dtype=torch.float64)
            Y2 = torch.tensor(samples_model.doa_p, dtype=torch.float64)
            model_dataset.append((X_model, Y1, Y2))
            generic_dataset.append((X, Y1, Y2))

    elif model_type.startswith("Trans2DSubNet"):
        for i in tqdm(range(samples_size)):
            # Samples model creation
            num = system_model_params.M
            snr = system_model_params.snr  # snr_list[i % len(snr_list)]
            T = T_list[i % len(T_list)]

            samples_model.set_doa(true_doa, num)

            # Observations matrix creation
            X = torch.tensor(samples_model.samples_creation(source_num=num, noise_mean=0, noise_variance=1,
                                                            signal_mean=0, signal_variance=1, snr=snr, snapshot=T)[0],
                             dtype=torch.complex64)
            X_model = create_autocorrelation_tensor(X, tau).to(torch.float)
            # Ground-truth creation
            Y1 = torch.tensor(samples_model.doa_t, dtype=torch.float64)
            Y2 = torch.tensor(samples_model.doa_p, dtype=torch.float64)
            model_dataset.append((X_model, Y1, Y2))
            generic_dataset.append((X, Y1, Y2))

    elif model_type.startswith("Classifier"):
        num_sources = [2, 3, 4, 5]
        for i in tqdm(range(samples_size)):
            num = num_sources[i % len(num_sources)]
            snr = system_model_params.snr   # snr_list[i % len(snr_list)]
            T = system_model_params.T  # T_list[i % len(T_list)]
            samples_model.set_doa(true_doa, num)
            X = torch.tensor(samples_model.samples_creation(source_num=num, noise_mean=0, noise_variance=1,
                                                            signal_mean=0, signal_variance=1, snr=snr, snapshot=T)[0],
                             dtype=torch.complex64)
            X_model = create_autocorrelation_tensor(X, tau).to(torch.float)
            Y = num
            # generic_dataset.append((X, Y))
            model_dataset.append((X_model, Y))
    else:
        for i in tqdm(range(samples_size)):
            num = system_model_params.M  # num_sources[i % len(num_sources)]
            # snr = snr_list[i % len(snr_list)]
            # Samples model creation
            samples_model.set_doa(true_doa, num)
            T = system_model_params.T
            # Observations matrix creation
            X = torch.tensor(samples_model.samples_creation(source_num=num, noise_mean=0, noise_variance=1,signal_mean=0,
                                                            signal_variance=1, snr=system_model_params.snr, snapshot=T)[0],
                             dtype=torch.complex64)
            if model_type.startswith(("SubspaceNet", "TransSubspaceNet")):
                # Generate auto-correlation tensor
                X_model = create_autocorrelation_tensor(X, tau).to(torch.float)
            elif model_type.startswith("DeepCNN") and phase.startswith("test"):
                # Generate 3d covariance parameters tensor
                X_model = create_cov_tensor(X)
            else:
                X_model = X
            # Ground-truth creation
            Y = torch.tensor(samples_model.doa, dtype=torch.float64)
            generic_dataset.append((X, Y))
            model_dataset.append((X_model, Y))

    if save_datasets:
        model_dataset_filename = f"{model_type}_DataSet_{system_model_params.signal_type}_" + \
                                 f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_" + \
                                 f"Nx={system_model_params.Nx}_Ny={system_model_params.Ny}_T={system_model_params.T}_" + \
                                 f"SNR={system_model_params.snr}_eta={system_model_params.eta}_" +\
                                 f"sv_noise_var{system_model_params.sv_noise_var}" + '.h5'
        generic_dataset_filename = f"Generic_DataSet_{system_model_params.signal_type}_" + \
                                   f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_" + \
                                   f"Nx={system_model_params.Nx}_Ny={system_model_params.Ny}_T={system_model_params.T}_" + \
                                   f"SNR={system_model_params.snr}_eta={system_model_params.eta}_" + \
                                   f"sv_noise_var{system_model_params.sv_noise_var}" + '.h5'
        samples_model_filename = f"samples_model_{system_model_params.signal_type}_" + \
                                 f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_" + \
                                 f"Nx={system_model_params.Nx}_Ny={system_model_params.Ny}_T={system_model_params.T}_" + \
                                 f"SNR={system_model_params.snr}_eta={system_model_params.eta}_" +\
                                 f"sv_noise_var{system_model_params.sv_noise_var}" + '.h5'

        torch.save(obj=model_dataset, f=datasets_path / phase / model_dataset_filename)
        torch.save(obj=generic_dataset, f=datasets_path / phase / generic_dataset_filename)
        if phase.startswith("test"):
            torch.save(obj=samples_model, f=datasets_path / phase / samples_model_filename)

    return model_dataset, generic_dataset, samples_model


# def read_data(Data_path: str) -> torch.Tensor:
def read_data(path: str):
    """
    Reads data from a file specified by the given path.

    Args:
    -----
        path (str): The path to the data file.

    Returns:
    --------
        torch.Tensor: The loaded data.

    Raises:
    -------
        None

    Examples:
    ---------
        >>> path = "data.pt"
        >>> read_data(path)

    """
    assert (isinstance(path, (str, Path)))
    data = torch.load(path)
    return data


# def autocorrelation_matrix(X: torch.Tensor, lag: int) -> torch.Tensor:
def autocorrelation_matrix(X: torch.Tensor, lag: int):
    '''
    Computes the autocorrelation matrix for a given lag of the input samples.

    Args:
    -----
        X (torch.Tensor): Samples matrix input with shape [N, T].
        lag (int): The requested delay of the autocorrelation calculation.

    Returns:
    --------
        torch.Tensor: The autocorrelation matrix for the given lag.

    '''
    Rx_lag = torch.zeros(X.shape[0], X.shape[0], dtype=torch.complex128).to(device)
    for t in range(X.shape[-1] - lag):
        # meu = torch.mean(X,1)
        x1 = torch.unsqueeze(X[:, t], 1).to(device)
        x2 = torch.t(torch.unsqueeze(torch.conj(X[:, t + lag]), 1)).to(device)
        dim = 0
        Rx_lag += torch.matmul(x1 - torch.mean(X), x2 - torch.mean(X)).to(device)

    Rx_lag = Rx_lag / (X.shape[-1] - lag)
    Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), dim)
    return Rx_lag


# def create_autocorrelation_tensor(X: torch.Tensor, tau: int) -> torch.Tensor:
def create_autocorrelation_tensor(X: torch.Tensor, tau: int):
    '''
    Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.
    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (BS, N, T).
        tau (int): Maximal time difference for the autocorrelation tensor.
    Returns:
    --------
        torch.Tensor: Tensor containing all the autocorrelation matrices,
                    with size (Batch size, tau, 2N, N).
    Raises:
    -------
        None
    '''
    Rx_tau = []
    for i in range(tau):
        Rx_tau.append(autocorrelation_matrix(X, lag=i))
    Rx_autocorr = torch.stack(Rx_tau, dim=0)
    return Rx_autocorr


# def create_cov_tensor(X: torch.Tensor) -> torch.Tensor:
def create_cov_tensor(X: torch.Tensor):
    '''
    Creates a 3D tensor of size (NxNx3) containing the real part, imaginary part, and phase component of the covariance matrix.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (N, T).

    Returns:
    --------
        Rx_tensor (torch.Tensor): Tensor containing the auto-correlation matrices, with size (Batch size, N, N, 3).

    Raises:
    -------
        None

    '''
    Rx = torch.cov(X)
    Rx_tensor = torch.stack((torch.real(Rx), torch.imag(Rx), torch.angle(Rx)), 2)
    return Rx_tensor


def load_datasets(system_model_params: SystemModelParams, model_type: str,
                  samples_size: float, train_test_ratio: float, datasets_path: Path, is_training: bool = False):
    """
    Load different datasets based on the specified parameters and phase.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        model_type (str): The type of the model.
        signal_type (str): The signal_type of the datasets.
        datasets_path (Path): The path to the datasets.
        is_training (bool): Specifies whether to load the training datasets.

    Returns:
    --------
        List: A list containing the loaded datasets.

    """
    datasets = []
    # Generate datasets filenames
    model_dataset_filename = f"{model_type}_DataSet_{system_model_params.signal_type}_" + \
                             f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_" + \
                             f"Nx={system_model_params.Nx}_Ny={system_model_params.Ny}_T={system_model_params.T}_" + \
                             f"SNR={system_model_params.snr}_eta={system_model_params.eta}_" + \
                             f"sv_noise_var{system_model_params.sv_noise_var}" + '.h5'
    generic_train_dataset_filename = f"Generic_DataSet_{system_model_params.signal_type}_" + \
                                     f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_" + \
                                     f"Nx={system_model_params.Nx}_Ny={system_model_params.Ny}_T={system_model_params.T}_" + \
                                     f"SNR={system_model_params.snr}_eta={system_model_params.eta}_" + \
                                     f"sv_noise_var{system_model_params.sv_noise_var}" + '.h5'
    model_test_dataset_filename = f"{model_type}_DataSet_{system_model_params.signal_type}_" + \
                                  f"{system_model_params.signal_nature}_{int(samples_size*train_test_ratio)}_" + \
                                  f"M={system_model_params.M}_Nx={system_model_params.Nx}_Ny={system_model_params.Ny}_" + \
                                  f"T={system_model_params.T}_SNR={system_model_params.snr}_eta={system_model_params.eta}_" + \
                                  f"sv_noise_var{system_model_params.sv_noise_var}" + '.h5'
    generic_test_dataset_filename = f"Generic_DataSet_{system_model_params.signal_type}_" + \
                                    f"{system_model_params.signal_nature}_{int(samples_size * train_test_ratio)}_" + \
                                    f"M={system_model_params.M}_Nx={system_model_params.Nx}_Ny={system_model_params.Ny}_" + \
                                    f"T={system_model_params.T}_SNR={system_model_params.snr}_eta={system_model_params.eta}_" + \
                                    f"sv_noise_var{system_model_params.sv_noise_var}" + '.h5'
    samples_test_model_filename = f"samples_model_{system_model_params.signal_type}_" + \
                                  f"{system_model_params.signal_nature}_{int(samples_size*train_test_ratio)}_" + \
                                  f"M={system_model_params.M}_Nx={system_model_params.Nx}_Ny={system_model_params.Ny}_" + \
                                  f"T={system_model_params.T}_SNR={system_model_params.snr}_eta={system_model_params.eta}_" + \
                                  f"sv_noise_var{system_model_params.sv_noise_var}" + '.h5'

    # Whether to load the training datasets
    if is_training:
        # Load training datasets
        try:
            train_dataset = read_data(datasets_path / "train" / model_dataset_filename)
            generic_train_dataset = read_data(datasets_path / "train" / generic_train_dataset_filename)
            datasets.append(train_dataset)
            datasets.append(generic_train_dataset)
        except:
            raise Exception("load_datasets: Training datasets doesn't exist")
    # Load test datasets
    try:
        test_dataset = read_data(datasets_path / "test" / model_test_dataset_filename)
        datasets.append(test_dataset)
    except:
        raise Exception("load_datasets: Test datasets doesn't exist")
    # Load generic test datasets
    try:
        generic_test_dataset = read_data(datasets_path / "test" / generic_test_dataset_filename)
        datasets.append(generic_test_dataset)
    except:
        raise Exception("load_datasets: Generic test datasets doesn't exist")
    # Load samples models
    try:
        samples_model = read_data(datasets_path / "test" / samples_test_model_filename)
        datasets.append(samples_model)
    except:
        raise Exception("load_datasets: Samples model datasets doesn't exist")
    return datasets
