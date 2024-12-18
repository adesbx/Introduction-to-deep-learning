# this file implement LeNet5 version with gridsearch hyperparameter tuning

import torch
import csv
from torch import Tensor
from torch.nn import Module, MSELoss, CrossEntropyLoss, Linear, Conv2d
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from core import core
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter()


class Cnn(Module):
    """Convolutional neural network

    Args:
                    Module (nn.Module): basic class of neural network
    """

    def __init__(self, output_nbr: int):
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


def hyper_param_tuning(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    loss_func: MSELoss | CrossEntropyLoss,
    batch_size_values: list[int],
    eta_values: list[float],
) -> Module:
    """grid search for tune best hyperparameter

    Args:
            train_dataset (TensorDataset):
            val_dataset (TensorDataset):
            loss_func (MSELoss | CrossEntropyLoss): function to minimize
            batch_size_values (list[int]):
            eta_values (list[float]): learning rate

    Returns:
            Module: best model
    """
    models = []
    for batch_size in batch_size_values:
        model = Cnn(10)
        for eta in eta_values:
            optim = torch.optim.SGD(model.parameters(), lr=eta)
            start_time = time.time()
            model_trained, nb_epoch, local_loss_mean, accuracy = (
                core.training_early_stopping(
                    model, train_dataset, val_dataset, batch_size, loss_func, optim
                )
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            model_info = [
                model_trained,
                accuracy,
                local_loss_mean,
                elapsed_time,
                batch_size,
                eta,
                nb_epoch,
            ]
            models.append(model_info)
            with open("./csv/dataCnn.csv", "a", newline="") as csvfile:
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
    best_model = max(models, key=lambda x: x[1])
    print("Meilleur hyper-paramètre \n", best_model[1:])
    return best_model[0]


if __name__ == "__main__":
    core = core()
    train_dataset, val_dataset, test_dataset = core.load_split_data()
    loss_func = torch.nn.MSELoss(reduction="mean")
    best_model = hyper_param_tuning(
        train_dataset, val_dataset, loss_func, [9], [0.01, 0.1]
    )
    core.final_test(best_model, test_dataset)
