# this file implement shallowNet version with grid-search hyper-parameter tuning

import torch
import csv
from torch import Tensor
from torch.nn import Module, MSELoss, CrossEntropyLoss, Linear
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from core import core
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter()


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


def hyper_param_tuning(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    loss_func: MSELoss | CrossEntropyLoss,
    batch_size_values: list[int],
    hidden_neuron_values: list[int],
    eta_values: list[float],
) -> Module:
    """grid search for tune best hyperparameter

    Args:
            train_dataset (TensorDataset):
            val_dataset (TensorDataset):
            loss_func (MSELoss | CrossEntropyLoss): function to minimize
            batch_size_values (list[int]): batch pick in future data_loader
            hidden_neuron_values (list[int]): number of neurons in the hidden layer
            eta_values (list[float]): learning rate

    Returns:
            Module: The best model
    """

    models = []
    for batch_size in batch_size_values:
        for hidden_neuron in hidden_neuron_values:
            model = ShallowNet(784, hidden_neuron, 10)
            for eta in eta_values:
                optim = torch.optim.Adam(model.parameters(), lr=eta)
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
                    hidden_neuron,
                    eta,
                    nb_epoch,
                ]
                models.append(model_info)
                with open("dataV2.csv", "a", newline="") as csvfile:
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
                        "lr": eta,
                        "batch_size": batch_size,
                        "hidden_neurons": hidden_neuron,
                        "nb_epoch": nb_epoch,
                    },
                    {
                        "hparam/Accuracy": accuracy,
                        "hparam/Validation Loss": local_loss_mean,
                        "hparam/time": elapsed_time,
                    },
                )

    best_model = min(models, key=lambda x: x[2])
    print("Meilleur hyper-paramètre \n", best_model[1:])
    return best_model[0]


if __name__ == "__main__":
    core = core()
    train_dataset, val_dataset, test_dataset = core.load_split_data()
    loss_func = torch.nn.MSELoss(reduction="mean")
    best_model = hyper_param_tuning(
        train_dataset, val_dataset, loss_func, [10], [1500], [0.0001]
    )
    core.final_test(best_model, test_dataset)
