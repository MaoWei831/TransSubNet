import math
import numpy as np
import torch
import torch.nn as nn
import warnings
from src.utils import gram_diagonal_overload
from src.utils import sum_of_diags_torch, find_roots_torch
from src.criterions import permute_prediction

warnings.simplefilter("ignore")
# Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepRootMUSIC(nn.Module):
    """DeepRootMUSIC is model-based deep learning model for DOA estimation problem.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        LeakyReLU (nn.LeakyReLU): Leaky reLu activation function, with activation_value.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, activation_value: float):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            activation_value (float): Value for the activation function.

        """
        super(DeepRootMUSIC, self).__init__()
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=2)
        self.LeakyReLU = nn.LeakyReLU(activation_value)
        self.DropOut = nn.Dropout(0.2)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the DeepRootMUSIC.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.BATCH_SIZE = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.LeakyReLU(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.LeakyReLU(x)
        # DCNN block #1
        x = self.deconv1(x)
        x = self.LeakyReLU(x)
        # DCNN block #2
        x = self.deconv2(x)
        x = self.LeakyReLU(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv3(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, :self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(Kx_tag, eps=1)  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to Root-MUSIC algorithm
        doa_prediction, doa_all_predictions, roots, _ = root_music(Rz, self.M, self.BATCH_SIZE)
        return doa_prediction, doa_all_predictions, roots, Rz


# TODO: inherit SubspaceNet from DeepRootMUSIC
class SubspaceNet(nn.Module):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        ReLU (nn.ReLU): ReLU activation function.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, M: int):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            M (int): Number of sources.

        """
        super(SubspaceNet, self).__init__()
        self.M = M
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(0.2)
        self.ReLU = nn.ReLU()

    def anti_rectifier(self, X):
        """Applies the anti-rectifier operation to the input tensor.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: Output tensor after applying the anti-rectifier operation.

        """
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.BATCH_SIZE = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.anti_rectifier(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #1
        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        # DCNN block #2
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, :self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(Kx=Kx_tag, eps=1, batch_size=self.BATCH_SIZE)  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to Root-MUSIC algorithm
        doa_prediction, doa_all_predictions, roots, _ = root_music(Rz, self.M, self.BATCH_SIZE)
        return doa_prediction, doa_all_predictions, roots, Rz


class TransSubspaceNet(SubspaceNet):

    def __init__(self, tau: int, M: int, N: int, T: int):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            M (int): Number of sources.

        """
        super(SubspaceNet, self).__init__()
        self.M = M
        self.N = N
        self.tau = min(tau, T)
        self.T = T
        self.BN = nn.BatchNorm1d(2*self.N**2)
        # self.pos_encoder = PositionalEncoding(2 * self.N**2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=2 * self.N**2,
                                                        nhead=1,
                                                        dim_feedforward=512,
                                                        dropout=0.1)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.conv1 = nn.Conv2d(self.tau, 1, kernel_size=3, padding=1, stride=1)
        self.fc = nn.Linear(self.tau, 1)

    def forward(self, Rx_tau: torch.Tensor):
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.BATCH_SIZE = Rx_tau.shape[0]  # [batchsize, tau, 2N, N]
        ## Architecture flow ###
        x = torch.reshape(Rx_tau, shape=(self.BATCH_SIZE, self.tau, self.N * 2 * self.N))
        x = x.permute(0, 2, 1).float().to(device)
        x = self.BN(x)
        # x = x.permute(2, 0, 1).float().to(device)  # [tau, batchsize, 2NN]
        # x = self.pos_encoder(x).to(device)
        x = x.permute(0, 2, 1).float().to(device)  # [batchsize, tau, 2NN]
        x = self.transformer(x)

        # x = x.permute(0, 2, 1).float().to(device)
        # x = self.fc(x)
        # Rx = torch.reshape(x, shape=(self.BATCH_SIZE, self.N * 2, self.N))
        x = torch.reshape(x, shape=(self.BATCH_SIZE, self.tau, self.N * 2, self.N))
        Rx = self.conv1(x)  # [batchsize, 1, 2N, N]
        Rx = Rx.squeeze(1)

        # Real and Imaginary Reconstruction
        Rx_real = Rx[:, :self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx[:, self.N:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(Kx=Kx_tag, eps=1, batch_size=self.BATCH_SIZE)  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to Root-MUSIC algorithm
        doa_prediction, doa_all_predictions, roots, _ = root_music(Rz, self.M, self.BATCH_SIZE)
        return doa_prediction, doa_all_predictions, roots, Rz


class Trans2DSubNet(nn.Module):
    def __init__(self, tau: int, M: int, Nx: int, Ny: int):
        super(Trans2DSubNet, self).__init__()
        self.M = M
        self.Nx = Nx
        self.Ny = Ny
        self.tau = tau
        # self.BatchNorm = nn.BatchNorm1d(2*(self.Nx+self.Ny)**2)  # 2 * (self.Nx + self.Ny)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=2 * (self.Ny + self.Nx) ** 2,
                                                        nhead=8,
                                                        dim_feedforward=2048)  # 512
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # [B, T, 2NxNy]  --->  [B, 1, 2NxNx]
        self.fc_x = nn.Linear(in_features=2 * (self.Ny + self.Nx) ** 2, out_features=2 * (self.Nx) ** 2)
        self.fc_y = nn.Linear(in_features=2 * (self.Ny + self.Nx) ** 2, out_features=2 * (self.Ny) ** 2)
        #         self.relu = nn.ReLU()  # 激活函数
        #         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.M * 2, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.M * 2),
        )

    def forward(self, Rx_tau: torch.Tensor):
        # X shape: [batchSize, NxNy, T]
        self.BATCH_SIZE = Rx_tau.shape[0]  # [batchSize, NxNy, T]

        # ### Architecture flow ###
        # [batchsize, T, 2NxNy]
        x = torch.reshape(Rx_tau, shape=(self.BATCH_SIZE, -1, 2 * (self.Ny + self.Nx) ** 2))

        # [batchsize, T, 2NxNy]
        x = self.transformer(x)
        x = torch.mean(x, dim=1)

        Rx = self.fc_x(x)
        #         Rx = self.relu(Rx)
        #         Rx = self.pool(Rx)
        Rx = torch.reshape(Rx.squeeze(), shape=(self.BATCH_SIZE, 2 * self.Nx, self.Nx))
        Ry = self.fc_y(x)
        #         Ry = self.relu(Ry)
        #         Ry = self.pool(Ry)
        Ry = torch.reshape(Ry.squeeze(), shape=(self.BATCH_SIZE, 2 * self.Ny, self.Ny))

        # Real and Imaginary Reconstruction
        Rx_real = Rx[:, :self.Nx, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx[:, self.Nx:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        Ry_real = Ry[:, :self.Ny, :]  # Shape: [Batch size, N, N])
        Ry_imag = Ry[:, self.Ny:, :]  # Shape: [Batch size, N, N])
        Ky_tag = torch.complex(Ry_real, Ry_imag)

        # Apply Gram operation diagonal loading
        Rz_x = gram_diagonal_overload(Kx=Kx_tag, eps=1, batch_size=self.BATCH_SIZE)
        Rz_y = gram_diagonal_overload(Kx=Ky_tag, eps=1, batch_size=self.BATCH_SIZE)  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to Root-MUSIC algorithm
        doa_t_pred, _, _, _ = root_music(Rz_x, self.M, self.BATCH_SIZE)
        doa_p_pred, _, _, _ = root_music(Rz_y, self.M, self.BATCH_SIZE)

        outs = self.fc(torch.cat((doa_t_pred, doa_p_pred), dim=1).float().to(device))

        return outs[:, :self.M], outs[:, self.M:]


class Trans2DMUSIC(nn.Module):
    def __init__(self, tau: int, M: int, Nx: int, Ny: int, T: int):
        super(Trans2DMUSIC, self).__init__()
        self.M = M
        self.Nx = Nx
        self.Ny = Ny
        self.tau = min(tau, T - 1)
        self.T = T  # snapshot
        self.BatchNorm = nn.BatchNorm1d(2 * self.Nx * self.Ny)  # 2 * self.Nx * self.Ny
        # x方向上的trans_encoder
        self.pos_encoder = PositionalEncoding(2 * (self.Ny * self.Nx))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=2 * (self.Ny * self.Nx),
                                                        nhead=8,
                                                        dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # [B, T, 2NxNy]  --->  [B, 1, 2NxNx]
        self.conv_x = nn.Conv1d(in_channels=self.T, out_channels=1, kernel_size=3, padding=1)
        self.conv_y = nn.Conv1d(in_channels=self.T, out_channels=1, kernel_size=3, padding=1)

        self.output = nn.Sequential(
            nn.Linear(in_features=360, out_features=64),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=64, out_features=6)
        ).to(device)

        self.ReLU = nn.ReLU()

    def anti_rectifier(self, X):
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def forward(self, X: torch.Tensor):
        # X shape: [batchSize, NxNy, T]
        self.BATCH_SIZE = X.shape[0]  # [batchSize, NxNy, T]
        # split real and imag
        x = torch.concat([X.real, X.imag], dim=1)  # [batchSize, 2NxNy, T]
        x = self.BatchNorm(x)

        x = x.permute(2, 0, 1).float().to(device)
        # [T, batchsize, 2NxNy]
        x = self.pos_encoder(x).to(device)
        x = x.permute(1, 0, 2).float().to(device)
        # [batchsize, T, 2NxNy]
        x = self.transformer(x)

        Rx = self.conv_x(x)
        Rx = torch.reshape(Rx.squeeze(), shape=(self.BATCH_SIZE, 2*self.Nx, self.Nx))
        Ry = self.conv_y(x)
        Ry = torch.reshape(Ry.squeeze(), shape=(self.BATCH_SIZE, 2*self.Ny, self.Ny))

        # Real and Imaginary Reconstruction
        Rx_real = Rx[:, :self.Nx, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx[:, self.Nx:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        Ry_real = Ry[:, :self.Ny, :]  # Shape: [Batch size, N, N])
        Ry_imag = Ry[:, self.Ny:, :]  # Shape: [Batch size, N, N])
        Ky_tag = torch.complex(Ry_real, Ry_imag)  # Shape: [Batch size, N, N])

        # Apply Gram operation diagonal loading
        spec_x = calculate_spectrum(Kx_tag, self.Nx).to(device)
        spec_y = calculate_spectrum(Ky_tag, self.Ny).to(device)

        doa_t_pred = self.output(spec_x)
        doa_p_pred = self.output(spec_y)

        return doa_t_pred, doa_p_pred


class SignalNum_Classifier(nn.Module):
    def __init__(self, N: int, tau: int):
        super(SignalNum_Classifier, self).__init__()

        # self.conv = nn.Conv2d(in_channels=tau, out_channels=1, kernel_size=3, padding=1)

        self.inputs = 2 * tau * N**2
        self.feedforward_dim = N**2
        self.outputs = N

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.inputs, out_features=self.feedforward_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.feedforward_dim, out_features=self.feedforward_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.feedforward_dim, out_features=self.feedforward_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.feedforward_dim, out_features=self.feedforward_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.feedforward_dim, out_features=self.outputs),
        )

    def forward(self, Rx: torch.Tensor):
        # Rx = self.conv(Rx)
        x = torch.reshape(Rx, (Rx.shape[0], -1))
        x = self.classifier(x)
        pred_num = torch.argmax(x, dim=1)

        return [x, pred_num]


class DeepAugmentedMUSIC(nn.Module):
    """ DeepAugmentedMUSIC is a model-based deep learning model for Direction of Arrival (DOA) estimation.

    Attributes:
        N (int): Number of sensors.
        T (int): Number of observations.
        M (int): Number of sources.
        angels (torch.Tensor): Tensor containing angles from -pi/2 to pi/2.
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden layer.
        rnn (nn.GRU): Recurrent neural network module.
        fc (nn.Linear): Fully connected layer.
        fc1 (nn.Linear): Fully connected layer.
        fc2 (nn.Linear): Fully connected layer.
        fc3 (nn.Linear): Fully connected layer.
        ReLU (nn.ReLU): Rectified Linear Unit activation function.
        DropOut (nn.Dropout): Dropout layer.
        BatchNorm (nn.BatchNorm1d): Batch normalization layer.
        sv (torch.Tensor): Steering vector.

    Methods:
        steering_vec(): Computes the steering vector based on the specified parameters.
        spectrum_calculation(Un: torch.Tensor): Calculates the MUSIC spectrum.
        pre_MUSIC(Rz: torch.Tensor): Applies the MUSIC operation for generating the spectrum.
        forward(X: torch.Tensor): Performs the forward pass of the DeepAugmentedMUSIC model.
    """

    def __init__(self, N: int, T: int, M: int):
        """Initializes the DeepAugmentedMUSIC model.

        Args:
        -----
            N (int): Number of sensors.
            M (int): Number of sources.
            T (int): Number of observations.
        """
        super(DeepAugmentedMUSIC, self).__init__()
        self.N, self.T, self.M = N, T, M
        self.angels = torch.linspace(-1 * np.pi / 2, np.pi / 2, 361)
        self.input_size = 2 * self.N
        self.hidden_size = 2 * self.N
        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size * self.N)
        self.fc1 = nn.Linear(self.angels.shape[0], self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.M)
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(0.25)
        self.BatchNorm = nn.BatchNorm1d(self.T)
        self.sv = self.steering_vec()
        # Weights initialization
        nn.init.xavier_uniform(self.fc.weight)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)

    def steering_vec(self):
        """Computes the ideal steering vector based on the specified parameters.
            equivalent to src.system_model.steering_vec method, but support pyTorch.

        Returns:
        --------
            tensor.Torch: the steering vector
        """
        sv = []
        for angle in self.angels:
            a = torch.exp(-1 * 1j * np.pi * torch.linspace(0, self.N - 1, self.N) * np.sin(angle))
            sv.append(a)
        return torch.stack(sv, dim=0)

    def spectrum_calculation(self, Un: torch.Tensor):

        spectrum_equation = []
        for i in range(self.angels.shape[0]):
            spectrum_equation.append(torch.real(torch.conj(self.sv[i]).T @ Un @ torch.conj(Un).T @ self.sv[i]))
        spectrum_equation = torch.stack(spectrum_equation, dim=0)
        spectrum = 1 / spectrum_equation

        return spectrum, spectrum_equation

    def pre_MUSIC(self, Rz: torch.Tensor):
        """Applies the MUSIC operration for generating spectrum

        Args:
            Rz (torch.Tensor): Generated covariance matrix

        Returns:
            torch.Tensor: The generated MUSIC spectrum
        """
        spectrum = []
        bs_Rz = Rz
        for iter in range(self.BATCH_SIZE):
            R = bs_Rz[iter]
            # Extract eigenvalues and eigenvectors using EVD
            _, eigenvectors = torch.linalg.eig(R)
            # Noise subspace as the eigenvectors which associated with the M first eigenvalues
            Un = eigenvectors[:, self.M:]
            # Calculate MUSIC spectrum
            spectrum.append(self.spectrum_calculation(Un)[0])
        return torch.stack(spectrum, dim=0)

    def forward(self, X: torch.Tensor):
        """
        Performs the forward pass of the DeepAugmentedMUSIC model.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: The estimated DOA.
        """
        # X shape == [Batch size, N, T]
        self.BATCH_SIZE = X.shape[0]
        ## Architecture flow ##
        # decompose X and concatenate real and imaginary part
        X = torch.cat((torch.real(X), torch.imag(X)), 1)  # Shape ==  [Batch size, 2N, T]
        # Reshape Output shape: [Batch size, T, 2N]
        X = X.view(X.size(0), X.size(2), X.size(1))
        # Apply batch normalization 
        X = self.BatchNorm(X)
        # GRU Clock
        gru_out, hn = self.rnn(X)
        Rx = gru_out[:, -1]
        # Reshape Output shape: [Batch size, 1, 2N]
        Rx = Rx.view(Rx.size(0), 1, Rx.size(1))
        # FC Block 
        Rx = self.fc(Rx)  # Shape: [Batch size, 1, 2N^2])
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_view = Rx.view(self.BATCH_SIZE, 2 * self.N, self.N)
        Rx_real = Rx_view[:, :self.N, :]  # Shape [Batch size, N, N])
        Rx_imag = Rx_view[:, self.N:, :]  # Shape [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape [Batch size, N, N])
        # Build MUSIC spectrum
        spectrum = self.pre_MUSIC(Kx_tag)  # Shape [Batch size, 361(grid_size)])
        # Apply peak detection using FC block #2
        y = self.ReLU(self.fc1(spectrum))  # Shape [Batch size, 361(grid_size)])
        y = self.ReLU(self.fc2(y))  # Shape [Batch size, 2N])
        y = self.ReLU(self.fc2(y))  # Shape [Batch size, 2N)
        # Find doa
        DOA = self.fc3(y)  # Shape [Batch size, M)
        return DOA


class DeepCNN(nn.Module):
    """DeepCNN is a convolutional neural network model for DoA  estimation.

    Args:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.

    Attributes:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.
        conv1 (nn.Conv2d): Convolutional layer 1.
        conv2 (nn.Conv2d): Convolutional layer 2.
        fc1 (nn.Linear): Fully connected layer 1.
        BatchNorm (nn.BatchNorm2d): Batch normalization layer.
        fc2 (nn.Linear): Fully connected layer 2.
        fc3 (nn.Linear): Fully connected layer 3.
        fc4 (nn.Linear): Fully connected layer 4.
        DropOut (nn.Dropout): Dropout layer.
        Sigmoid (nn.Sigmoid): Sigmoid activation function.
        ReLU (nn.ReLU): Rectified Linear Unit activation function.

    Methods:
        forward(X: torch.Tensor): Performs the forward pass of the DeepCNN model.
    """

    def __init__(self, N, grid_size):
        ## input dim (N, T)
        super(DeepCNN, self).__init__()
        self.N = N
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2)
        self.fc1 = nn.Linear(256 * (self.N - 5) * (self.N - 5), 4096)
        self.BatchNorm = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, self.grid_size)
        self.DropOut = nn.Dropout(0.3)
        self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, X):
        # X shape == [Batch size, N, N, 3]
        X = X.view(X.size(0), X.size(3), X.size(2), X.size(1))  # [Batch size, 3, N, N]
        ## Architecture flow ##
        # CNN block #1: 3xNxN-->256x(N-2)x(N-2)
        X = self.conv1(X)
        X = self.ReLU(X)
        # CNN block #2: 256x(N-2)x(N-2)-->256x(N-3)x(N-3)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #3: 256x(N-3)x(N-3)-->256x(N-4)x(N-4)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #4: 256x(N-4)x(N-4)-->256x(N-5)x(N-5)
        X = self.conv2(X)
        X = self.ReLU(X)
        # FC BLOCK
        # Reshape Output shape: [Batch size, 256 * (self.N - 5) * (self.N - 5)]
        X = X.view(X.size(0), -1)
        X = self.DropOut(self.ReLU(self.fc1(X)))  # [Batch size, 4096]
        X = self.DropOut(self.ReLU(self.fc2(X)))  # [Batch size, 2048]
        X = self.DropOut(self.ReLU(self.fc3(X)))  # [Batch size, 1024]
        X = self.fc4(X)  # [Batch size, grid_size]
        X = self.Sigmoid(X)
        return X


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]

        return self.dropout(x)


def root_music(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Root-MUSIC algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix. it is equal to
        src.methods: RootMUSIC.narrowband() method. 

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
        doa_all_batches (torch.Tensor): All doa predicted, given all roots, over all batches.
        roots_to_return (torch.Tensor): The unsorted roots.
    """

    dist = 0.5
    f = 1
    doa_batches = []
    doa_all_batches = []
    roots_batchs = []
    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)
        # Assign noise subspace as the eigenvectors associated with M greatest eigenvalues 
        Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:]
        # Generate hermitian noise subspace matrix 
        F = torch.matmul(Un, torch.t(torch.conj(Un)))
        # Calculates the sum of F matrix diagonals
        diag_sum = sum_of_diags_torch(F)
        # Calculates the roots of the polynomial defined by F matrix diagonals
        roots = find_roots_torch(diag_sum)
        # Calculate the phase component of the roots
        roots_angels_all = torch.angle(roots)
        # Calculate doa
        doa_pred_all = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels_all)
        doa_all_batches.append(doa_pred_all)
        roots_to_return = roots
        # Take only roots which inside the unit circle
        roots = roots[sorted(range(roots.shape[0]), key=lambda k: abs(abs(roots[k]) - 1))]
        mask = (torch.abs(roots) - 1) < 0
        roots = roots[mask][:M]
        # Calculate the phase component of the roots
        roots_angels = torch.angle(roots)  # 默认为弧度制
        roots_angels = (1 / (2 * np.pi * dist * f)) * roots_angels
        roots_batchs.append(roots_angels)
        # Calculate doa
        doa_pred = torch.arcsin(roots_angels)
        doa_batches.append(doa_pred)

    return torch.stack(doa_batches, dim=0), torch.stack(doa_all_batches, dim=0), \
        roots_to_return, torch.stack(roots_batchs, dim=0)


snapshots = 200


def ULA_action_vector(array, theta):
    return np.exp(- 1j * np.pi * array * np.sin(theta))


def calculate_spectrum(En, m):
    array = np.linspace(0, m, m, endpoint=False)  # array element positions
    angles = np.array((np.linspace(- np.pi / 2, np.pi / 2, 360, endpoint=False),))
    r = angles.shape[1]
    a = torch.zeros([m, r]) + 1j * torch.zeros([m, r])
    for i in range(r):
        a[:, i] = torch.from_numpy(ULA_action_vector(array, angles[0, i]))
    a = torch.complex(a.real.float(), a.imag.float()).to(device)

    H1 = torch.matmul(En.to(device) @ torch.conj(En.permute(0, 2, 1)).to(device), a).to(device)
    H2 = torch.mul(H1, torch.conj(a))
    H3 = torch.sum(H2, dim=1)

    return (1.0 / abs(H3)).to(device)
