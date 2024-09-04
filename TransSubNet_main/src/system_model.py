import numpy as np


class SystemModelParams:
    """Class for setting parameters of a system model."""

    def __init__(self):
        """
        Initialize the SystemModelParams object.

        Parameters:
            None

        Attributes:
            M (int): Number of sources.
            N (int): Number of sensors.
            T (int): Number of observations.
            signal_type (str): Signal type ("NarrowBand" or "Broadband").
            freq_values (list): Frequency values for Broadband signal.
            signal_nature (str): Signal nature ("non-coherent" or "coherent").
            snr (float): Signal-to-noise ratio.
            eta (float): Sensor location deviation.
            sv_noise_var (float): Steering vector added noise variance.

        Returns:
            None
        """
        self.M = None  # Number of sources
        self.N = None  # Number of sensors
        self.Nx = None
        self.Ny = None
        self.T = None  # Number of observations
        self.signal_type = None  # Signal type ("NarrowBand" or "Broadband")
        self.freq_values = [0, 500]  # Frequency values for Broadband signal
        self.signal_nature = None  # Signal nature ("non-coherent" or "coherent")
        self.snr = None  # Signal-to-noise ratio
        self.eta = None  # Sensor location deviation
        self.sv_noise_var = None  # Steering vector added noise variance
        self.is_onebit = None  #
        self.signal_dim = None  # Signal Dimension
        self.mixed = None

    def set_num_sources(self, M: int):
        """
        Set the number of sources.
        """
        self.M = M
        return self

    def set_num_sensors(self, N: int):
        """
        Set the number of sensors.
        """
        self.N = N
        self.Nx = N
        self.Ny = N
        return self

    def set_num_observations(self, T: int):
        """
        Set the number of observations.
        """
        self.T = T
        return self

    def set_signal_type(self, signal_type: str, freq_values: list = [0, 500]):
        """
        Set the signal type.

        Parameters:
            signal_type (str): Signal type ("NarrowBand" or "Broadband").
            freq_values (list, optional): Frequency values for Broadband signal.
                Defaults to [0, 500].

        """
        self.signal_type = signal_type
        if signal_type.startswith("Broadband"):
            self.freq_values = freq_values
        return self

    def set_signal_nature(self, signal_nature: str):
        """
        Set the signal nature.

        Parameters:
            signal_nature (str): Signal nature ("non-coherent" or "coherent").

        """
        self.signal_nature = signal_nature
        return self

    def set_snr(self, snr: float):
        """
        Set the signal-to-noise ratio.
        """
        self.snr = snr
        return self

    def set_sensors_dev(self, eta: float):
        """
        Set the level of deviation from sensor location.
        """
        self.eta = eta
        return self

    def set_sv_noise(self, sv_noise_var: float):
        """
        Set the steering vector added noise variance.
        """
        self.sv_noise_var = sv_noise_var
        return self

    def set_array_shape(self, set_array_shape: str):
        self.set_array_shape = set_array_shape
        return self

    def set_dataset_mixed(self, mixed: bool):
        self.mixed = mixed
        return self


class SystemModel(object):
    def __init__(self, system_model_params: SystemModelParams):
        """Class used for defining the settings of the system model.

        Attributes:
        -----------
            signal_type (str): Signals type. Options: "NarrowBand", "Broadband".
            N (int): Number of sensors.
            M (int): Number of sources.
            freq_values (list, optional): Frequency range for broadband signals. Defaults to None.
            min_freq (dict): Minimal frequency value for different scenarios.
            max_freq (dict): Maximal frequency value for different scenarios.
            f_rng (dict): Frequency range of interest for different scenarios.
            f_sampling (dict): Sampling rate for different scenarios.
            time_axis (dict): Time axis for different scenarios.
            dist (dict): Distance between array elements for different scenarios.
            array (np.ndarray): Array of sensor locations.

        Methods:
        --------
            define_scenario_params(freq_values: list): Defines the signal_type parameters.
            create_array(): Creates the array of sensor locations.
            steering_vec(theta: np.ndarray, f: float = 1, array_form: str = "ULA",
                eta: float = 0, geo_noise_var: float = 0) -> np.ndarray: Computes the steering vector.

        """
        self.params = system_model_params
        # Assign signal type parameters  
        self.define_scenario_params()
        # Define array indices 
        self.create_array()

    def define_scenario_params(self):
        """Defines the signal type parameters based on the specified frequency values.
        """
        freq_values = self.params.freq_values
        # Define minimal frequency value
        self.min_freq = {"NarrowBand": None, "Broadband": freq_values[0]}
        # Define maximal frequency value
        self.max_freq = {"NarrowBand": None, "Broadband": freq_values[1]}
        # Frequency range of interest
        self.f_rng = {"NarrowBand": None,
                      "Broadband": np.linspace(start=self.min_freq["Broadband"], stop=self.max_freq["Broadband"], \
                                               num=self.max_freq["Broadband"] - self.min_freq["Broadband"],
                                               endpoint=False)}
        # Define sampling rate as twice the maximal frequency
        self.f_sampling = {"NarrowBand": None,
                           "Broadband": 2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"])}
        # Define time axis   
        self.time_axis = {"NarrowBand": None,
                          "Broadband": np.linspace(0, 1, self.f_sampling["Broadband"], endpoint=False)}
        # distance between array elements
        self.dist = {"NarrowBand": 1 / 2,
                     "Broadband": 1 / (2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"]))}

    def create_array(self):
        """ create an array of sensors locations
        """
        if self.params.set_array_shape.startswith('ULA'):
            self.array = np.linspace(0, self.params.N, self.params.N, endpoint=False)
        else:
            self.x_array = np.linspace(0, self.params.Nx, self.params.Nx, endpoint=False)
            self.y_array = np.linspace(0, self.params.Ny, self.params.Ny, endpoint=False)

    def steering_vec(self, theta: np.ndarray, phi: np.ndarray = None, f: float = 1, array_form="ULA"):
        """Computes the steering vector based on the specified parameters.

        Args:
        -----
            theta (np.ndarray): Array of angles.
            f (float, optional): Frequency. Defaults to 1.
            array_form (str, optional): Array form. Defaults to "ULA".

        Returns:
        --------
            np.ndarray: Computed steering vector.

        """
        f_sv = {"NarrowBand": 1, "Broadband": f}
        if array_form.startswith("ULA"):
            # define uniform deviation in spacing (for each sensor)
            mis_distance = np.random.uniform(low=-1 * self.params.eta, high=self.params.eta, size=self.params.N)
            # define noise added to steering vector
            mis_geometry_noise = np.sqrt(self.params.sv_noise_var) * (np.random.randn(self.params.N))
            return np.exp(-2 * 1j * np.pi * f_sv[self.params.signal_type] *
                          (mis_distance + self.dist[self.params.signal_type]) * self.array * np.sin(theta)) + \
                mis_geometry_noise

        elif array_form.startswith("UPA"):
            # define uniform deviation in spacing (for each sensor)
            mis_distance_x = np.random.uniform(low=-1 * self.params.eta, high=self.params.eta, size=self.params.Nx)
            mis_distance_y = np.random.uniform(low=-1 * self.params.eta, high=self.params.eta, size=self.params.Ny)
            # define noise added to steering vector
            mis_geometry_x_noise = np.sqrt(self.params.sv_noise_var) * (np.random.randn(self.params.Nx))
            mis_geometry_y_noise = np.sqrt(self.params.sv_noise_var) * (np.random.randn(self.params.Ny))
            Ax = np.exp(-2 * 1j * np.pi * f_sv[self.params.signal_type] *
                        (mis_distance_x + self.dist[self.params.signal_type]) *
                        self.x_array * np.sin(theta) * np.cos(phi)) + mis_geometry_x_noise
            Ay = np.exp(-2 * 1j * np.pi * f_sv[self.params.signal_type] *
                        (mis_distance_y + self.dist[self.params.signal_type]) *
                        self.y_array * np.sin(theta) * np.sin(phi)) + mis_geometry_y_noise
            return np.kron(Ay, Ax)

        elif array_form.startswith("L-shaped"):
            # define uniform deviation in spacing (for each sensor)
            mis_distance_x = np.random.uniform(low=-1 * self.params.eta, high=self.params.eta, size=self.params.Nx)
            mis_distance_y = np.random.uniform(low=-1 * self.params.eta, high=self.params.eta, size=self.params.Ny)
            # define noise added to steering vector
            mis_geometry_x_noise = np.sqrt(self.params.sv_noise_var) * (np.random.randn(self.params.Nx))
            mis_geometry_y_noise = np.sqrt(self.params.sv_noise_var) * (np.random.randn(self.params.Ny))
            Ax = np.exp(-2 * 1j * np.pi * f_sv[self.params.signal_type] *
                        (mis_distance_x + self.dist[self.params.signal_type]) *
                        self.x_array * np.sin(phi) * np.cos(theta)) + mis_geometry_x_noise
            Ay = np.exp(-2 * 1j * np.pi * f_sv[self.params.signal_type] *
                        (mis_distance_y + self.dist[self.params.signal_type]) *
                        self.y_array * np.sin(phi) * np.sin(theta)) + mis_geometry_y_noise
            return [Ax, Ay]
        else:
            raise Exception(f"SystemModel.steering_vec: array form {array_form} is not defined")

    def __str__(self):
        """Returns a string representation of the SystemModel object.
        """
        if self.params.signal_type.startwith("Broadband"):
            print("System Model Summery:")
            for key, value in self.__dict__.items():
                print(key, " = ", value)
            return "End of Model"
