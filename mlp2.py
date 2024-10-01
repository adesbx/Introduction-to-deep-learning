import torch
import csv
import optuna
from optuna.trial import TrialState
import torch.nn as nn
import torch.nn.functional as F
from core import core
from torch.utils.tensorboard import SummaryWriter
import time
writer = SummaryWriter()
core = core()

class Mlp(nn.Module):

    def __init__(self, layers_nbr, layers_in_out):
        super(Mlp, self).__init__()
        self.layers_nbr = layers_nbr
        self.hidden_layers = nn.ModuleList()
        for n in range(layers_nbr-1):
            print(layers_in_out[n])
            self.hidden_layers.append(nn.Linear(layers_in_out[n][0], layers_in_out[n][1]))
        print(layers_in_out[layers_nbr-1])
        self.output = nn.Linear(layers_in_out[layers_nbr-1][0],
                                layers_in_out[layers_nbr-1][1])
    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output(x)
        return x
 
def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 2, 10)
    layers_in_out = [(784, trial.suggest_int("n_layer1", 200, 1000))]
    n_output = layers_in_out[0][1] #output of the first layer in the case we have 2 layer
    for n in range(n_layers-2):
        n_output = trial.suggest_int(f'nlayer{n}', 200, 1000)
        layers_in_out.append(
            (layers_in_out[n][1],
             n_output)
        )
    layers_in_out.append((n_output, 10))
    model = Mlp(n_layers, layers_in_out)
    return model

def objective(trial):
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
    loss_func =  torch.nn.MSELoss()
    # Training of the model.
    start_time = time.time()
    model_trained, nb_epoch ,local_loss_mean, accuracy = core.training_early_stopping(model, train_dataset, val_dataset, batch_size, loss_func, optimizer, trial)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # List of out_features for hidden_layers
    hidden_out_features = [layer.out_features for layer in model.hidden_layers]
    # TODO write good informations
    model_info = [model_trained, accuracy, local_loss_mean, elapsed_time, batch_size, hidden_out_features, model.layers_nbr, lr, nb_epoch]
    trial.set_user_attr(key="model", value=model_trained)
    with open('./csv/dataMlp2.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        if csvfile.tell() == 0:
            spamwriter.writerow(['Accuracy', 'Validation Loss', 'Elapsed time', 'Batch Size', 'Hidden layer', 'Hidden Num', 'Learning Rate', 'Epochs'])
        spamwriter.writerow(model_info[1:])
    writer.add_hparams(
        {'lr': lr, 'batch_size': batch_size, 'hidden_neurons': str(hidden_out_features), 'layers': model.layers_nbr,'nb_epoch': nb_epoch},
        {'hparam/Accuracy': accuracy, 'hparam/Validation Loss': local_loss_mean, 'hparam/time': elapsed_time}
    )
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=1000)

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
