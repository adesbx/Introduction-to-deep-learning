# this file implement LeNet5 version with gridsearch hyperparameter tuning

from optuna import Trial
import optuna
import torch
import csv
from torch import Tensor
from optuna.trial import TrialState
from torch.nn import Module, Linear, Conv2d
import torch.nn.functional as F
from core import core
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter()


class Cnn(Module):
    """Convolutional neural network

    Args:
                                    Module (nn.Module): basic class of neural network
    """

    def __init__(
        self,
        output_nbr: int,
    ):
        """Constructor

        Args:
                        output_nbr (int): number of neurons in the output layer
        """
        super(Cnn, self).__init__()
        self.conv1 = Conv2d(1, 6, kernel_size=(5, 5))
        self.conv2 = Conv2d(6, 16, kernel_size=(5, 5))

        self.fc1 = Linear(16 * 4 * 4, 120)
        self.fc2 = Linear(120, 84)

        self.output = Linear(84, output_nbr)

    def forward(self, x: Tensor) -> Tensor:
        """process of a neuron in the model

        Args:
                        x (Tensor): the batch of data selected

        Returns:
                        Tensor: output of the neuron
        """
        x = x.view(-1, 1, 28, 28)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.output(x)

        return output


def define_model(trial: Trial) -> Module:
    """define the model, for hyperparameter number of neuron

    Args:
            trial (Trial): Object for select number of neuron

    Returns:
            Module: model configured
    """
    model = Cnn(10)
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
        lr,
        nb_epoch,
    ]
    trial.set_user_attr(key="model", value=model_trained)
    with open("./csv/dataMlp2.csv", "a", newline="") as csvfile:
        spamwriter = csv.writer(csvfile)
        if csvfile.tell() == 0:
            spamwriter.writerow(
                [
                    "Accuracy",
                    "Validation Loss",
                    "Elapsed time",
                    "Batch Size",
                    "Learning Rate",
                    "Epochs",
                ]
            )
        spamwriter.writerow(model_info[1:])
    writer.add_hparams(
        {
            "lr": lr,
            "batch_size": batch_size,
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
    core = core()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000,  timeout=10800)

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
