# Imports
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.system_model import SystemModelParams
# from src.signal_creation import *
from src.data_handler import create_dataset, load_datasets
from src.criterions import set_criterions
from src.training import TrainingParams, simulation_summary, train
from src.evaluation import evaluate_dnn_model, evaluate
from src.plotting import initialize_figures
from src.utils import *
from pathlib import Path
from datetime import datetime

# Initialization
warnings.simplefilter("ignore")
os.system('cls||clear')
plt.close('all')

# Initialize paths
external_data_path = Path.cwd() / "data"
scenario_data_path = "LowSNR"
datasets_path = external_data_path / "datasets" / scenario_data_path
simulations_path = external_data_path / "simulations"
saving_path = external_data_path / "weights/final_models"

# Initialize time and date
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")


def run(commands, system_model_params, model_type, simulation_filename, trained_filename):
    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)

    set_unified_seed()  # Initialize seed

    # ########################  Datasets creation  ########################
    if commands["CREATE_DATA"]:
        set_unified_seed()
        # Define which datasets to generate
        create_training_data = True
        create_testing_data = True
        print("Creating Data...")
        if create_training_data:
            # Generate training datasets
            # 这里的数据集已经做过自相关处理
            train_dataset, generic_train_dataset, _ = create_dataset(system_model_params=system_model_params,
                                                                     samples_size=samples_size,
                                                                     tau=tau,
                                                                     model_type=model_type,
                                                                     datasets_path=datasets_path,
                                                                     phase="train")
        if create_testing_data:
            # Generate test datasets
            test_dataset, generic_test_dataset, samples_model = create_dataset(system_model_params=system_model_params,
                                                                               samples_size=int(train_test_ratio * samples_size),
                                                                               tau=tau,
                                                                               model_type=model_type,
                                                                               datasets_path=datasets_path,
                                                                               phase="test")

    # ########################  Datasets loading   ###########################
    if commands["LOAD_DATA"]:
        train_dataset, generic_train_dataset, \
            test_dataset, generic_test_dataset, samples_model = load_datasets(system_model_params=system_model_params,
                                                                              model_type=model_type,
                                                                              samples_size=samples_size,
                                                                              train_test_ratio=train_test_ratio,
                                                                              datasets_path=datasets_path,
                                                                              is_training=True)

    # ########################  Training stage   ##############################
    if commands["TRAIN_MODEL"]:
        # Assign the training parameters object
        simulation_parameters = TrainingParams(model_type=model_type) \
            .set_batch_size(32).set_epochs(100) \
            .set_model(system_model=samples_model, tau=tau) \
            .set_optimizer(optimizer="Adam", learning_rate=5e-4, weight_decay=1e-8) \
            .set_training_dataset(train_dataset) \
            .set_schedular(step_size=5, gamma=0.9) \
            .set_criterion()
        # train_dataset是已经完成相关度计算的数据，generic为原始数据

        if commands["LOAD_MODEL"]:
            simulation_parameters.load_model(loading_path=saving_path / trained_filename)

        # Print training simulation details
        simulation_summary(system_model_params=system_model_params, model_type=model_type,
                           parameters=simulation_parameters, phase="training")

        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(training_parameters=simulation_parameters,
                                                        model_name=simulation_filename, saving_path=saving_path)
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(model.state_dict(), saving_path / "final_models" / Path(simulation_filename))
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(simulations_path / "results" / "plots" / Path(dt_string_for_save + r".png"))
        else:
            plt.show()

    # ########################  Evaluation stage  #############################
    if commands["EVALUATE_MODE"]:
        # Initialize figures dict for plotting
        figures = initialize_figures()
        # Define loss measure for evaluation
        criterion, subspace_criterion = set_criterions("rmse")
        # Load datasets for evaluation
        if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
            test_dataset, generic_test_dataset, samples_model = load_datasets(
                system_model_params=system_model_params, model_type=model_type,
                samples_size=samples_size, train_test_ratio=train_test_ratio,
                datasets_path=datasets_path)
        # Generate DataLoader objects
        model_test_dataset = torch.utils.data.DataLoader(test_dataset,
                                                         batch_size=1, shuffle=False, drop_last=False)
        generic_test_dataset = torch.utils.data.DataLoader(generic_test_dataset,
                                                           batch_size=1, shuffle=False, drop_last=False)
        # Load pre-trained model
        if not commands["TRAIN_MODEL"]:
            # Define an evaluation parameters instance
            simulation_parameters = TrainingParams(model_type="SubspaceNet") \
                .set_model(system_model=samples_model, tau=tau) \
                .load_model(loading_path=saving_path / simulation_filename) \
                .set_criterion()
            model = simulation_parameters.model

        # print simulation summary details
        simulation_summary(system_model_params=system_model_params, model_type=model_type,
                           phase="evaluation", parameters=simulation_parameters)
        # Evaluate DNN models, augmented and subspace methods

        # model_test_loss = evaluate_dnn_model(model=model, dataset=model_test_dataset,
        #                                      criterion=criterion,
        #                                      model_type=model_type)

        evaluate(model=model, model_type="SubspaceNet", model_test_dataset=model_test_dataset,
                 generic_test_dataset=generic_test_dataset, criterion=criterion,
                 subspace_criterion=subspace_criterion, system_model=samples_model,
                 figures=figures, plot_spec=True)

        # loss, acc = evaluate_classifier(model=model, dataset=model_test_dataset,
        #                                 criterion=simulation_parameters.criterion)
        # print('loss = ', loss, ', acc = ', acc)


if __name__ == "__main__":
    for i in range(2):
        # Operations commands
        commands = {"SAVE_TO_FILE": False,  # Saving results to file or present them over CMD
                    "CREATE_DATA": True,  # Creating new data
                    "LOAD_DATA": False,  # Loading data from datasets
                    "LOAD_MODEL": False,  # Load specific model
                    "TRAIN_MODEL": False,  # Applying training operation
                    "SAVE_MODEL": False,  # Saving tuned model
                    "EVALUATE_MODE": False}  # Evaluating desired algorithms
        # Saving simulation scores to external file

        if commands["SAVE_TO_FILE"]:
            file_path = simulations_path / "results" / Path("scores" + dt_string_for_save + ".txt")
            sys.stdout = open(file_path, "w")

        # ########################  Datasets creation  ########################
        # Define system model parameters
        system_model_params = SystemModelParams() \
            .set_num_sensors(8) \
            .set_num_sources(3+i) \
            .set_num_observations(100) \
            .set_snr(10) \
            .set_signal_type("NarrowBand") \
            .set_signal_nature("coherent") \
            .set_sensors_dev(eta=0) \
            .set_sv_noise(0) \
            .set_array_shape("L-shaped")

        # Set model type, options:
        # DA-MUSIC, DeepCNN ,Classifier, Trans2DSubNet，TransSubspaceNet， SubspaceNet
        model_type = "Trans2DSubNet"
        tau = 8
        samples_size = 50000  # Overall dateset size
        train_test_ratio = 0.1  # training and testing datasets ratio

        # Sets Saved Model filename
        simulation_filename = f"{model_type}_M={system_model_params.M}_" + \
                              f"T={system_model_params.T}_SNR_{system_model_params.snr}_" + \
                              f"tau={tau}_{system_model_params.signal_type}_" + \
                              f"{system_model_params.signal_nature}_eta={system_model_params.eta}_" + \
                              f"sv_noise={system_model_params.sv_noise_var}"

        trained_filename = "SubspaceNet_M=2_SNR_0_tau=8_NarrowBand_coherent_mixT"

        run(commands, system_model_params, model_type, simulation_filename, simulation_filename)
