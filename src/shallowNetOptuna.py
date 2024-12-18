# This file implement automatic hyper parameter tuning of ShallowNet

from optuna import Trial
import torch
import optuna
from optuna.trial import TrialState
import csv
import os
from torch import Tensor
from torch.nn import Module, MSELoss, CrossEntropyLoss, Linear
import torch.nn.functional as F
from core import core
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter()
core = core()
test_dataset_g = None


class ShallowNet(Module):
    """The ShallowNet class extend of Module from pytorch

    Args:
        Module (nn.Module): basic class of neural network model
    """

    def __init__(self, input_num: int, hidden_num: int, output_num: int):
        """constructor

        Args:
                input_num (int): number of data input
                hidden_num (int): number of neuron in the hidden layer
                output_num (int): number of neuron in the output layer
        """
        super(ShallowNet, self).__init__()
        self.hidelayer1 = Linear(input_num, hidden_num)
        self.output = Linear(hidden_num, output_num)

    def forward(self, x: Tensor) -> Tensor:
        """process of a neuron in the model

        Args:
                x (Tensor): the batch of data selected

        Returns:
                Tensor: output of the neuron
        """
        x = self.hidelayer1(x)
        x = F.relu(x)
        x = self.output(x)
        return x


def define_model(trial: Trial) -> Module:
    """define the model, for hyperparameter number of neuron

    Args:
        trial (Trial): Object for select number of neuron

    Returns:
        Module: model configured
    """
    # We optimize the hidden units.
    hidden_neuron = trial.suggest_int("n_units_hidenlayer", 500, 2000)
    model = ShallowNet(784, hidden_neuron, 10)
    return model


def objective(trial: Trial) -> float:
    """function to maximise

    Args:
        trial (Trial): For suggest hyperparameter

    Returns:
        float: accuracy of the model tuned
    """
    # Generate the model.
    model = define_model(trial)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    train_dataset, val_dataset, test_dataset = core.load_split_data()
    global test_dataset_g
    test_dataset_g = test_dataset
    batch_size = trial.suggest_int("batch", 3, 100)
    loss_func = torch.nn.MSELoss()
    # Training of the model.
    start_time = time.time()
    model_trained, nb_epoch, local_loss_mean, accuracy = core.training_early_stopping(
        model, train_dataset, val_dataset, batch_size, loss_func, optimizer, trial
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    model_info = [
        model_trained,
        accuracy,
        local_loss_mean,
        elapsed_time,
        batch_size,
        model.hidelayer1.out_features,
        lr,
        nb_epoch,
    ]
    trial.set_user_attr(key="model", value=model_trained)
    with open("./csv/dataV2.csv", "a", newline="") as csvfile:
        spamwriter = csv.writer(csvfile)
        if csvfile.tell() == 0:
            spamwriter.writerow(
                [
                    "Accuracy",
                    "Validation Loss",
                    "Elapsed time",
                    "Batch Size",
                    "Hidden Num",
                    "Learning Rate",
                    "Epochs",
                ]
            )
        spamwriter.writerow(model_info[1:])
    writer.add_hparams(
        {
            "lr": lr,
            "batch_size": batch_size,
            "hidden_neurons": model.hidelayer1.out_features,
            "nb_epoch": nb_epoch,
        },
        {
            "hparam/Accuracy": accuracy,
            "hparam/Validation Loss": local_loss_mean,
            "hparam/time": elapsed_time,
        },
    )
    return accuracy


if __name__ == "__main__":
    # set_pytorch_multicore()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    core.final_test(trial.user_attrs["model"], test_dataset_g)