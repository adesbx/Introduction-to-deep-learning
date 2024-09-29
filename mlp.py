import torch
import csv
import torch.nn as nn
import torch.nn.functional as F
from core import core
from torch.utils.tensorboard import SummaryWriter
import time
writer = SummaryWriter()


class Mlp(nn.Module):

	def __init__(self, input_nbr, hidden_layers_nbr, hidden_neuron_nbr, output_nbr):
		super(Mlp, self).__init__()
		self.hidden_layers = nn.ModuleList([nn.Linear(input_nbr, hidden_neuron_nbr)])
		for n in range(hidden_layers_nbr-1):
			self.hidden_layers.append(nn.Linear(hidden_neuron_nbr, hidden_neuron_nbr))
		self.output = nn.Linear(hidden_neuron_nbr, output_nbr)

	def forward(self, x):
		for layer in self.hidden_layers: 
			x = layer(x)
			x = F.relu(x)
		x = self.output(x)
		return x


def hyper_param_tuning(train_dataset, val_dataset, loss_func, batch_size_values, hidden_layers_values, hidden_neuron_values, eta_values):
	models = []
	for batch_size in batch_size_values:
		for hidden_neuron in hidden_neuron_values:
			for hidden_layer in hidden_layers_values:
				model = Mlp(784, hidden_layer, hidden_neuron, 10)
				for eta in eta_values:
					optim = torch.optim.SGD(model.parameters(), lr=eta)
					start_time = time.time()
					model_trained, nb_epoch ,local_loss_mean, accuracy = core.training_early_stopping(model, train_dataset, val_dataset, batch_size,
							loss_func, optim)
					end_time = time.time()
					elapsed_time = end_time - start_time
					model_info = [model_trained, accuracy, local_loss_mean, elapsed_time, batch_size, hidden_layer, hidden_neuron, eta, nb_epoch]
					models.append(model_info)
					with open('./csv/dataMlpV2.csv', 'a', newline='') as csvfile:
						spamwriter = csv.writer(csvfile)
						if csvfile.tell() == 0:
							spamwriter.writerow(['Accuracy', 'Validation Loss', 'Elapsed time', 'Batch Size', 'Hidden layer', 'Hidden Num', 'Learning Rate', 'Epochs'])
						spamwriter.writerow(model_info[1:])
	best_model = max(models, key=lambda x: x[1])
	print("Meilleur hyper-paramètre \n", best_model[1:])
	return best_model[0]

if __name__ == "__main__":
	core = core()
	train_dataset, val_dataset, test_dataset = core.load_split_data()
	loss_func = torch.nn.MSELoss(reduction='mean')
	best_model = hyper_param_tuning(train_dataset, val_dataset, loss_func, [9, 15, 30], [1, 5, 20, 50], [350, 500, 600], [0.05, 0.01, 0.1])
	core.final_test(best_model, test_dataset)