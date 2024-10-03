# this file implement grid-search mlp hyperparameter tuning
import torch
import csv
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.nn import Module, MSELoss, CrossEntropyLoss, Linear, ModuleList
import torch.nn.functional as F
from core import core
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter()


class Mlp(Module):
    """The Multi-layer Perceptron

    Args:
        Module (nn.Module):
    """

    def __init__(
        self,
        input_nbr: int,
        hidden_layers_nbr: int,
        hidden_neuron_nbr: int,
        output_nbr: int,
    ):
        """Constructor

        Args:
            input_nbr (int):
            hidden_layers_nbr (int):
            hidden_neuron_nbr (int):
            output_nbr (int):
        """
        super(Mlp, self).__init__()
        self.hidden_layers = ModuleList([Linear(input_nbr, hidden_neuron_nbr)])
        for n in range(hidden_layers_nbr - 1):
            self.hidden_layers.append(Linear(hidden_neuron_nbr, hidden_neuron_nbr))
        self.output = Linear(hidden_neuron_nbr, output_nbr)

    def forward(self, x: Tensor) -> Tensor:
        """process of a neuron in the model

        Args:
                x (Tensor): the batch of data selected

        Returns:
                Tensor: output of the neuron
        """
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output(x)
        return x


def hyper_param_tuning(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    loss_func: MSELoss | CrossEntropyLoss,
    batch_size_values: list[int],
    hidden_layers_values: list[int],
    hidden_neuron_values: list[int],
    eta_values: list[float],
) -> Module:
    """grid search for tune best hyperparameter

    Args:
            train_dataset (TensorDataset):
            val_dataset (TensorDataset):
            loss_func (MSELoss | CrossEntropyLoss): function to minimize
            batch_size_values (list[int]): batch pick in future data_loader
            hidden_layers_values (list[int]): number of layer in the mlp
            hidden_neuron_values (list[int]): number of neurons in the hidden layer
            eta_values (list[float]): learning rate

    Returns:
            Module: The best model
    """
    models = []
    for batch_size in batch_size_values:
        for hidden_neuron in hidden_neuron_values:
            for hidden_layer in hidden_layers_values:
                model = Mlp(784, hidden_layer, hidden_neuron, 10)
                for eta in eta_values:
                    optim = torch.optim.SGD(model.parameters(), lr=eta)
                    start_time = time.time()
                    model_trained, nb_epoch, local_loss_mean, accuracy = (
                        core.training_early_stopping(
                            model,
                            train_dataset,
                            val_dataset,
                            batch_size,
                            loss_func,
                            optim,
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
                        hidden_layer,
                        hidden_neuron,
                        eta,
                        nb_epoch,
                    ]
                    models.append(model_info)
                    with open("./csv/dataMlpV2.csv", "a", newline="") as csvfile:
                        spamwriter = csv.writer(csvfile)
                        if csvfile.tell() == 0:
                            spamwriter.writerow(
                                [
                                    "Accuracy",
                                    "Validation Loss",
                                    "Elapsed time",
                                    "Batch Size",
                                    "Hidden layer",
                                    "Hidden Num",
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
        train_dataset,
        val_dataset,
        loss_func,
        [9, 15, 30],
        [1, 5, 20, 50],
        [350, 500, 600],
        [0.05, 0.01, 0.1],
    )
    core.final_test(best_model, test_dataset)
