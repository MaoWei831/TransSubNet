import numpy as np
from src.system_model import SystemModel, SystemModelParams
from src.utils import D2R


class Samples(SystemModel):
    """
    Class used for defining and creating signals and observations.
    Inherits from SystemModel class.

    ...

    Attributes:
    -----------
        doa (np.ndarray): Array of angels (directions) of arrival.

    Methods:
    --------
        set_doa(doa): Sets the direction of arrival (DOA) for the signals.
        samples_creation(noise_mean: float = 0, noise_variance: float = 1, signal_mean: float = 0,
            signal_variance: float = 1): Creates samples based on the specified mode and parameters.
        noise_creation(noise_mean, noise_variance): Creates noise based on the specified mean and variance.
        signal_creation(signal_mean=0, signal_variance=1, SNR=10): Creates signals based on the specified mode and parameters.
    """

    def __init__(self, system_model_params: SystemModelParams):
        """Initializes a Samples object.

        Args:
        -----
        system_model_params (SystemModelParams): an instance of SystemModelParams,
            containing all relevant system model parameters.

        """
        super().__init__(system_model_params)

    def set_doa(self, doa, M: int):
        """
        Sets the direction of arrival (DOA) for the signals.

        Args:
        -----
            doa (np.ndarray): Array containing the DOA values.

        """

        def create_doa_with_gap(gap: float, doa_range: int):

            while True:
                DOA = np.round(np.random.rand(M) * doa_range, decimals=2)  # - 90)
                DOA.sort()
                diff_angles = np.array([np.abs(DOA[i + 1] - DOA[i]) for i in range(M - 1)])
                if ((np.sum(diff_angles > gap) == M - 1) and (np.sum(diff_angles < (180 - gap)) == M - 1)):
                    break
            return DOA

        def create_doa_mixed(gap: float, doa_range: int, M: int):
            DOA = (np.round(np.random.rand(M) * doa_range, decimals=2) - 90)
            return DOA

        if doa == None:
            # Generate angels with gap greater than 0.2 rad (nominal case)
            # -pi/2-pi/2
            if self.params.set_array_shape.startswith("ULA"):
                self.doa = np.array(create_doa_with_gap(gap=15, doa_range=180)) * D2R
            # 0-pi, 0-pi/2
            elif self.params.set_array_shape.startswith(("UPA", "L-shaped")):
                self.doa_t = np.array(create_doa_with_gap(gap=15, doa_range=180)) * D2R
                self.doa_p = np.array(create_doa_with_gap(gap=15, doa_range=90)) * D2R
            else:
                raise Exception(f"Samples.samples_creation: signal dimension {self.params.signal_dim} is not defined")
        else:
            # Generate
            self.doa = np.array(doa)

    def samples_creation(self, source_num: int = 6, noise_mean: float = 0, noise_variance: float = 1,
                         signal_mean: float = 0, signal_variance: float = 1, snr: int = 0, snapshot=200):
        """ Creates samples based on the specified mode and parameters.

        Args:
        -----
            noise_mean (float, optional): Mean of the noise. Defaults to 0.
            noise_variance (float, optional): Variance of the noise. Defaults to 1.
            signal_mean (float, optional): Mean of the signal. Defaults to 0.
            signal_variance (float, optional): Variance of the signal. Defaults to 1.

        Returns:
        --------
            tuple: Tuple containing the created samples, signal, steering vectors, and noise.

        Raises:
        -------
            Exception: If the signal_type is not defined.

        """
        # Generate signal matrix
        signal = self.signal_creation(signal_mean, signal_variance, M=source_num, snr=snr, snapshot=snapshot)
        # Generate noise matrix
        noise = self.noise_creation(noise_mean, noise_variance, snapshot=snapshot)
        # Generate Narrowband samples
        if self.params.signal_type.startswith("NarrowBand"):
            # 创建二维数据，均匀面阵/L型阵列
            if self.params.set_array_shape.startswith("UPA"):
                A = np.array([self.steering_vec(theta=theta, phi=phi, array_form='UPA')
                              for theta, phi in zip(self.doa_t, self.doa_p)])
                samples = (A.T @ signal) + noise
                return samples, signal, A, noise

            if self.params.set_array_shape.startswith("L-shaped"):
                Ax = np.array([self.steering_vec(theta=theta, phi=phi, array_form='L-shaped')[0]
                              for theta, phi in zip(self.doa_t, self.doa_p)])
                Ay = np.array([self.steering_vec(theta=theta, phi=phi, array_form='L-shaped')[1]
                               for theta, phi in zip(self.doa_t, self.doa_p)])
                A = np.concatenate((Ax, Ay), axis=1)
                samples = np.concatenate((Ax.T @ signal, Ay.T @ signal)) + noise
                return samples, signal, A, noise

            A = np.array([self.steering_vec(theta) for theta in self.doa]).T
            samples = (A @ signal) + noise
            return samples, signal, A, noise

        # Generate Broadband samples
        elif self.params.signal_type.startswith("Broadband"):
            samples = []
            SV = []

            for idx in range(self.f_sampling["Broadband"]):

                # mapping from index i to frequency f
                if idx > int(self.f_sampling["Broadband"]) // 2:
                    f = - int(self.f_sampling["Broadband"]) + idx
                else:
                    f = idx
                A = np.array([self.steering_vec(theta, f) for theta in self.doa]).T
                samples.append((A @ signal[:, idx]) + noise[:, idx])
                SV.append(A)
            samples = np.array(samples)
            SV = np.array(SV)
            samples_time_domain = np.fft.ifft(samples.T, axis=1)[:, :snapshot]
            return samples_time_domain, signal, SV, noise
        else:
            raise Exception(f"Samples.samples_creation: signal type {self.params.signal_type} is not defined")

    def noise_creation(self, noise_mean, noise_variance, snapshot=200):
        """ Creates noise based on the specified mean and variance.

        Args:
        -----
            noise_mean (float): Mean of the noise.
            noise_variance (float): Variance of the noise.

        Returns:
        --------
            np.ndarray: Generated noise.

        """
        # for NarrowBand signal_type Noise represented in the time domain
        if self.params.signal_type.startswith("NarrowBand"):
            if self.params.set_array_shape.startswith("ULA"):
                return np.sqrt(noise_variance) * (np.sqrt(2) / 2) * \
                    (np.random.randn(self.params.N, snapshot) +
                     1j * np.random.randn(self.params.N, snapshot)) + noise_mean
            if self.params.set_array_shape.startswith("UPA"):
                return np.sqrt(noise_variance) * (np.sqrt(2) / 2) * \
                    (np.random.randn(self.params.Nx * self.params.Ny, snapshot) + \
                     1j * np.random.randn(self.params.Nx * self.params.Ny, snapshot)) + \
                    noise_mean
            if self.params.set_array_shape.startswith("L-shaped"):
                n = np.sqrt(noise_variance) * (np.sqrt(2) / 2) * \
                    (np.random.randn(self.params.Nx + self.params.Ny, snapshot) + \
                     1j * np.random.randn(self.params.Nx + self.params.Ny, snapshot)) + \
                    noise_mean
                n[self.params.Nx, :] = n[self.params.Nx-1, :]
                return n

        # for Broadband signal_type Noise represented in the frequency domain
        elif self.params.signal_type.startswith("Broadband"):
            noise = np.sqrt(noise_variance) * (np.sqrt(2) / 2) * (
                    np.random.randn(self.params.N, len(self.time_axis["Broadband"])) \
                    + 1j * np.random.randn(self.params.N, len(self.time_axis["Broadband"]))) + noise_mean
            return np.fft.fft(noise)

        else:
            raise Exception(f"Samples.noise_creation: signal type {self.params.signal_type} is not defined")

    def signal_creation(self, signal_mean: float = 0, signal_variance: float = 1,
                        M: int = None, snr: int = 0, snapshot: int = 200):
        """
        Creates signals based on the specified signal nature and parameters.

        Args:
        -----
            signal_mean (float, optional): Mean of the signal. Defaults to 0.
            signal_variance (float, optional): Variance of the signal. Defaults to 1.

        Returns:
        --------
            np.ndarray: Created signals.

        Raises:
        -------
            Exception: If the signal type is not defined.
            Exception: If the signal nature is not defined.
        """
        amplitude = (10 ** (snr / 10))
        # NarrowBand signal creation
        if self.params.signal_type == "NarrowBand":
            if self.params.signal_nature == "non-coherent":
                # create M non-coherent signals
                return amplitude * (np.sqrt(2) / 2) * np.sqrt(signal_variance) * \
                    (np.random.randn(M, snapshot)
                     + 1j * np.random.randn(M, snapshot)) + signal_mean

            elif self.params.signal_nature == "coherent":
                # Coherent signals: same amplitude and phase for all signals
                sig = amplitude * (np.sqrt(2) / 2) * np.sqrt(signal_variance) * \
                      (np.random.randn(1, snapshot) + 1j * np.random.randn(1, snapshot)) + signal_mean
                return np.repeat(sig, M, axis=0)

        # OFDM Broadband signal creation
        elif self.params.signal_type.startswith("Broadband"):
            num_sub_carriers = self.max_freq["Broadband"]  # number of subcarriers per signal
            if self.params.signal_nature == "non-coherent":
                # create M non-coherent signals
                signal = np.zeros((M, len(self.time_axis["Broadband"]))) + \
                         1j * np.zeros((M, len(self.time_axis["Broadband"])))
                for i in range(M):
                    for j in range(num_sub_carriers):
                        sig_amp = amplitude * (np.sqrt(2) / 2) * (np.random.randn(1) + 1j * np.random.randn(1))
                        signal[i] += sig_amp * np.exp(1j * 2 * np.pi * j * len(self.f_rng["Broadband"]) * \
                                                      self.time_axis["Broadband"] / num_sub_carriers)
                    signal[i] *= (1 / num_sub_carriers)
                return np.fft.fft(signal)
            # Coherent signals: same amplitude and phase for all signals
            elif self.params.signal_nature == "coherent":
                signal = np.zeros((1, len(self.time_axis["Broadband"]))) + \
                         1j * np.zeros((1, len(self.time_axis["Broadband"])))
                for j in range(num_sub_carriers):
                    sig_amp = amplitude * (np.sqrt(2) / 2) * (np.random.randn(1) + 1j * np.random.randn(1))
                    signal += sig_amp * np.exp(1j * 2 * np.pi * j * len(self.f_rng["Broadband"]) * \
                                               self.time_axis["Broadband"] / num_sub_carriers)
                signal *= (1 / num_sub_carriers)
                return np.tile(np.fft.fft(signal), (M, 1))
            else:
                raise Exception(f"signal nature {self.params.signal_nature} is not defined")

        else:
            raise Exception(f"signal type {self.params.signal_type} is not defined")
