import gzip
import torch
import optuna
from optuna import Trial
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module, MSELoss, CrossEntropyLoss

writer = SummaryWriter()


class core:
    """Implement generalisation methods, for all models"""

    def load_split_data(
        self, path: str = "mnist.pkl.gz", percentage: list[int] = [0.8, 0.2]
    ) -> tuple:
        """load from the path and split dataset in train, validation and test part

        Args:
            path (str, optional): path to the dataset. Defaults to "mnist.pkl.gz".
            percentage (list[int], optional): percentage of repartition for train and validation. Defaults to [0.8, 0.2].

        Returns:
            tuple: train, val, test dataset
        """
        ((data_train, label_train), (data_test, label_test)) = torch.load(
            gzip.open(path), weights_only=True
        )
        train_dataset = TensorDataset(data_train, label_train)
        train_dataset, val_dataset = random_split(train_dataset, percentage)
        test_dataset = TensorDataset(data_test, label_test)
        return train_dataset, val_dataset, test_dataset

    def train_step(
        self,
        model: Module,
        train_loader: DataLoader,
        loss_func: MSELoss | CrossEntropyLoss,
        optim: SGD | Adam,
        epoch: int,
    ):
        """train the model from train_loader, adjusted weight

        Args:
            model (Module): pytorch model
            train_loader (DataLoader): train loader
            loss_func (MSELoss | CrossEntropyLoss): goal is to min this function
            optim (SGD | Adam): optimise the gradient descent
            epoch (int): epoch of this train step for write the loss/train
        """
        # on lit toutes les données d'apprentissage
        model.train()
        running_loss = 0.0
        i = 0
        for x, t in train_loader:
            # on calcule la sortie du modèle
            y = model(x)
            # on met à jour les poids
            loss = loss_func(t, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            running_loss += loss.item()
            if i % 1000 == 999:  # every 1000 mini-batches...
                # ...log the running loss
                writer.add_scalar(
                    "loss/train", running_loss / 1000, epoch * len(train_loader) + i
                )
                running_loss = 0.0
            i += 1

    def validation_step(
        self,
        model: Module,
        val_loader: DataLoader,
        loss_func: MSELoss | Adam,
        epoch: int,
    ):
        """validate the model from val_loader, doesn't move weight

        Args:
            model (Module): pytorch model
            val_loader (DataLoader): validation loader
            loss_func (MSELoss | CrossEntropyLoss): goal is to min this function
            optim (SGD | Adam): optimise the gradient descent
            epoch (int): epoch of this train step for write the loss/train
        """
        loss_total = 0
        acc = 0.0
        total_samples = 0
        i = 0.0
        running_loss = 0.0
        model.eval()  # prep model for evaluation
        for x, t in val_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            y = model(x)
            # calculate the loss
            loss = loss_func(t, y)
            loss_total += loss
            # calculate the accuracy
            acc += (torch.argmax(y, 1) == torch.argmax(t, 1)).sum().item()
            total_samples += t.size(0)
            running_loss += loss.item()
            # record validation loss
            if i % 1000 == 999:  # every 1000 mini-batches...
                # ...log the running loss
                writer.add_scalar(
                    "loss/validation", running_loss / 1000, epoch * len(val_loader) + i
                )
                running_loss = 0.0
            i += 1
        return loss_total / len(val_loader), acc / total_samples

    def training_early_stopping(
        self,
        model: Module,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        batch_size: int,
        loss_func: MSELoss | CrossEntropyLoss,
        optim: MSELoss | Adam,
        trial: Trial = None,
        max_epochs: int = 100,
        min_delta: int = 0.0005,
        patience: int = 10,
    ) -> tuple:
        """training phase, with early-stopping if no more convergence

        Args:
            model (Module): pytorch model
            train_dataset (TensorDataset):
            val_dataset (TensorDataset):
            batch_size (int): size of batch that is pick in the dataloader
            loss_func (MSELoss | CrossEntropyLoss): function to minimize
            optim (MSELoss | Adam): optimise the gradient descent
            trial (Trial, optional): Object from optuna for manage the hyper parameter choice. Defaults to None.
            max_epochs (int, optional): limit of a train_step call. Defaults to 100.
            min_delta (int, optional): minimum of improvement before lossing patience . Defaults to 0.0005.
            patience (int, optional): decrease until 0 for early-stopping. Defaults to 10.

        Raises:
            optuna.exceptions.TrialPruned: Pruned by algo from optuna

        Returns:
            tuple: model, number of epoch, loss, accuracy
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        previous_loss_mean = 1
        previous_improvement = 1
        for n in range(max_epochs):
            self.train_step(model, train_loader, loss_func, optim, n)
            local_loss_mean, accuracy = self.validation_step(
                model, val_loader, loss_func, n
            )
            improvement = previous_loss_mean - local_loss_mean
            if previous_improvement == improvement or improvement <= min_delta:
                if patience == 0:
                    previous_loss_mean = local_loss_mean
                    print(
                        f"early-stopped at {n} epochs for {previous_loss_mean} loss_mean"
                    )
                    break
                patience -= 1
            previous_loss_mean = local_loss_mean
            previous_improvement = improvement
            if trial:
                trial.report(accuracy, n)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            writer.flush()
        return model, n, previous_loss_mean.item(), accuracy

    def final_test(self, best_model: Module, test_dataset: TensorDataset):
        """Final test for the best of the best model

        Args:
            best_model (Module): the best pytorch model trained
            test_dataset (TensorDataset):
        """
        acc = 0.0
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        # on lit toutes les donnéees de test
        for x, t in test_loader:
            # on calcule la sortie du modèle
            y = best_model(x)
            # on regarde si la sortie est correcte
            acc += torch.argmax(y, 1) == torch.argmax(t, 1)
        # on affiche le pourcentage de bonnes réponses
        print(acc / len(test_dataset))
