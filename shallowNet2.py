import torch
import csv
import os
import torch.nn as nn
import torch.nn.functional as F
from core import core
from torch.utils.tensorboard import SummaryWriter
import time
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

def hyper_param_tuning(train_dataset, val_dataset, loss_func, batch_size_values, hidden_neuron_values, eta_values):
	models = []
	for batch_size in batch_size_values:
		for hidden_neuron in hidden_neuron_values:
			model = ShallowNet(784, hidden_neuron, 10)
			for eta in eta_values:
				optim = torch.optim.Adam(model.parameters(), lr=eta)
				start_time = time.time()
				model_trained, nb_epoch ,local_loss_mean, accuracy = core.training_early_stopping(model, train_dataset, val_dataset, batch_size,
						loss_func, optim)
				end_time = time.time()
				elapsed_time = end_time - start_time
				model_info = [model_trained, accuracy, local_loss_mean, elapsed_time, batch_size, hidden_neuron, eta, nb_epoch]
				models.append(model_info)
				with open('dataV2.csv', 'a', newline='') as csvfile:
					spamwriter = csv.writer(csvfile)
					if csvfile.tell() == 0:
						spamwriter.writerow(['Accuracy', 'Validation Loss', 'Elapsed time','Batch Size', 'Hidden Num', 'Learning Rate', 'Epochs'])
					spamwriter.writerow(model_info[1:])
				writer.add_hparams(
                    {'lr': eta, 'batch_size': batch_size, 'hidden_neurons': hidden_neuron, 'nb_epoch': nb_epoch},
                    {'hparam/Accuracy': accuracy, 'hparam/Validation Loss': local_loss_mean, 'hparam/time': elapsed_time}
                )
                
	best_model = min(models, key=lambda x: x[2])		
	print("Meilleur hyper-paramètre \n", best_model[1:])
	return best_model[0]

def set_pytorch_multicore():
    num_cores = os.cpu_count()
    torch.set_num_threads(num_cores)
    
    print(f"PyTorch configuré pour utiliser {num_cores} cœurs CPU")

if __name__ == "__main__":
	core = core()
	# set_pytorch_multicore()
	train_dataset, val_dataset, test_dataset = core.load_split_data()
	loss_func = torch.nn.MSELoss(reduction='mean')
	best_model = hyper_param_tuning(train_dataset, val_dataset, loss_func, [10], [1500], [0.0001])
	core.final_test(best_model, test_dataset)
