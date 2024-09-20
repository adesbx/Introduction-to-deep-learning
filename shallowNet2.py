import torch
import csv
import torch.nn as nn
import torch.nn.functional as F
from core import core
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


class ShallowNet(nn.Module):

	def __init__(self, input_num, hidden_num, output_num):
		super(ShallowNet, self).__init__()
		self.hidelayer1 = nn.Linear(input_num, hidden_num)
		self.output = nn.Linear(hidden_num, output_num)

	def forward(self, x):
		x = self.hidelayer1(x)
		x = F.relu(x)
		x = self.output(x)
		return x

def load_split_data(path: str = 'mnist.pkl.gz', percentage: list() = [0.8, 0.2]):
	((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open(path))
	train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
	train_dataset, val_dataset = random_split(train_dataset, percentage)
	test_dataset = torch.utils.data.TensorDataset(data_test,label_test)
	return train_dataset, val_dataset, test_dataset

def train_step(model, train_loader, loss_func, optim):
	# on lit toutes les données d'apprentissage
	model.train()
	for x,t in train_loader:
		# on calcule la sortie du modèle
		y = model(x)
		# on met à jour les poids
		loss = loss_func(t,y)
		loss.backward()
		optim.step()
		optim.zero_grad()

def validation_step(model, val_loader, loss_func, shape):
	loss_total = 0
	acc = 0.
	model.eval() # prep model for evaluation
	for x, t in val_loader:
		# forward pass: compute predicted outputs by passing inputs to the model
		y = model(x)
		# calculate the loss
		loss = loss_func(t,y)
		loss_total += loss

		#calculate the accuracy
		acc += torch.argmax(y,1) == torch.argmax(t,1)

		# record validation loss
	return loss_total/len(val_loader), acc.mean()/shape

def training_early_stopping(model, train_dataset, val_dataset, batch_size, loss_func, optim, max_epochs=100, min_delta=0.001, patience=2):
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	previous_loss_mean = 1
	previous_improvement = 1
	for n in range(max_epochs):
		train_step(model, train_loader, loss_func, optim)
		local_loss_mean, accuracy = validation_step(model, val_loader, loss_func, len(val_dataset))
		improvement = previous_loss_mean-local_loss_mean
		if previous_improvement==improvement or improvement <= min_delta:
			if patience == 0:
				previous_loss_mean=local_loss_mean
				print(f"early-stopped at {n} epochs for {previous_loss_mean} loss_mean")
				break
			patience -= 1
		previous_loss_mean=local_loss_mean
		previous_improvement=improvement
		# écrire dans tensorboard
		writer.add_scalar("Loss", local_loss_mean, n)
		writer.add_scalar("Accuracy", accuracy, n)
	return model, n, previous_loss_mean.item(), accuracy

def hyper_param_tuning(train_dataset, val_dataset, loss_func, batch_size_values, hidden_neuron_values, eta_values):
	models = []
	for batch_size in batch_size_values:
		for hidden_neuron in hidden_neuron_values:
			model = ShallowNet(784, hidden_neuron, 10)
			for eta in eta_values:
				optim = torch.optim.SGD(model.parameters(), lr=eta)
				model_trained, nb_epoch ,local_loss_mean, accuracy = core.training_early_stopping(model, train_dataset, val_dataset, batch_size,
						loss_func, optim)
				model_info = [model_trained, local_loss_mean, batch_size, hidden_neuron, eta, nb_epoch]
				models.append(model_info)
				with open('data.csv', 'a', newline='') as csvfile:
					spamwriter = csv.writer(csvfile)
					if csvfile.tell() == 0:
						spamwriter.writerow(['Validation Loss', 'Batch Size', 'Hidden Num', 'Learning Rate', 'Epochs'])
					spamwriter.writerow(model_info[1:])

				writer.add_hparams(
                    {'lr': eta, 'batch_size': batch_size, 'hidden_neurons': hidden_neuron},
                    {'Accuracy': accuracy, 'Validation Loss': local_loss_mean}
                )
                
	best_model = min(models, key=lambda x: x[1])		
	print("Meilleur hyper-paramètre \n", best_model[1:])
	return best_model[0]

if __name__ == "__main__":
	core = core()
	train_dataset, val_dataset, test_dataset = core.load_split_data()
	loss_func = torch.nn.MSELoss(reduction='mean')
	best_model = hyper_param_tuning(train_dataset, val_dataset, loss_func, [3], [250], [0.001, 0.01])
	core.final_test(best_model, test_dataset)
