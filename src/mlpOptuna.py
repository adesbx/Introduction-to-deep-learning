# this file implement automatic hyperparameter tuning of mlp

from optuna import Trial
import torch
import csv
import optuna
from optuna.trial import TrialState
from torch import Tensor
from torch.nn import Module, Linear, ModuleList
import torch.nn.functional as F
from core import core
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter()
core = core()


class Mlp(Module):
    """The Multi-layer Perceptron

    Args:
        Module (nn.Module):
    """

    def __init__(self, layers_nbr: int, layers_in_out: list[tuple[int, int]]):
        """constructor

        Args:
            layers_nbr (int): number of layers
            layers_in_out (_type_): contain number of neurons in and out foreach layers
        """
        super(Mlp, self).__init__()
        self.layers_nbr = layers_nbr
        self.hidden_layers = ModuleList()
        for n in range(layers_nbr - 1):
            self.hidden_layers.append(Linear(layers_in_out[n][0], layers_in_out[n][1]))
        self.output = Linear(
            layers_in_out[layers_nbr - 1][0], layers_in_out[layers_nbr - 1][1]
        )

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


def define_model(trial: Trial) -> Module:
    """define the model, for hyperparameter number of neuron

    Args:
        trial (Trial): Object for select number of neuron

    Returns:
        Module: model configured
    """
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 2, 10)
    layers_in_out = [(784, trial.suggest_int("n_layer1", 200, 1000))]
    n_output = layers_in_out[0][
        1
    ]  # output of the first layer in the case we have 2 layer
    for n in range(n_layers - 2):
        n_output = trial.suggest_int(f"nlayer{n}", 200, 1000)
        layers_in_out.append((layers_in_out[n][1], n_output))
    layers_in_out.append((n_output, 10))
    model = Mlp(n_layers, layers_in_out)
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
    # List of out_features for hidden_layers
    hidden_out_features = [layer.out_features for layer in model.hidden_layers]
    model_info = [
        model_trained,
        accuracy,
        local_loss_mean,
        elapsed_time,
        batch_size,
        hidden_out_features,
        model.layers_nbr,
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
                    "Hidden layer",
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
            "hidden_neurons": str(hidden_out_features),
            "layers": model.layers_nbr,
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
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000, timeout=10800)

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
