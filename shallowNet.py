import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import csv

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

batch_size = 5 # nombre de données lues à chaque fois
nb_epochs = 10 # nombre de fois que la base de données sera lue
hidden_num = 250 # nombre de neurones cachés
eta = 0.00001 # taux d'apprentissage

loss_func = torch.nn.MSELoss(reduction='mean')
((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
test_dataset = torch.utils.data.TensorDataset(data_test,label_test)
train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])


models = []

for batch_size in [3, 5, 10]:
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
	for hidden_num in [150, 200, 250, 300]:
		model = ShallowNet(784, hidden_num, 10)
		for eta in [0.00001, 0.0001, 0.001, 0.01]:
			optim = torch.optim.SGD(model.parameters(), lr=eta)
			for nb_epochs in [5, 10, 20]:
				print('batch value: \n', batch_size)
				print('nb_epoch value: \n', nb_epochs)
				print('hidden_run: ', hidden_num)
				print('eta:', eta)
				# on initiliase l'optimiseur

				for n in range(nb_epochs):
					# on lit toutes les données d'apprentissage
					for x,t in train_loader:
						# on calcule la sortie du modèle
						y = model(x)
						# on met à jour les poids
						loss = loss_func(t,y)
						loss.backward()
						optim.step()
						optim.zero_grad()
				
					loss_total = 0
					model.eval() # prep model for evaluation
					for x, t in val_loader:
						# forward pass: compute predicted outputs by passing inputs to the model
						y = model(x)
						# calculate the loss
						loss = loss_func(t,y)
						loss_total += loss
						# record validation loss
					if n == nb_epochs-1:
						models.append([model, loss_total/len(val_loader), batch_size, hidden_num, eta, nb_epochs])
						with open('data.csv', 'a', newline='') as csvfile:
							spamwriter = csv.writer(csvfile)
							if csvfile.tell() == 0:
								spamwriter.writerow(['Validation Loss', 'Batch Size', 'Hidden Num', 'Learning Rate', 'Epochs'])
							spamwriter.writerow([(loss_total/len(val_loader)).item(), batch_size, hidden_num, eta, nb_epochs])
						# print('batch value: \n', batch_size)
						# print('nb_epoch value: \n', nb_epochs)
						# print('hidden_run: ', hidden_num)
						# print('eta:', eta)

bestModel = min(models, key=lambda x: x[1])		
model = bestModel[0]
print("Meilleur hyper-paramètre \n", bestModel)
acc = 0.
# on lit toutes les donnéees de test
for x,t in test_loader:
	# on calcule la sortie du modèle
	y = model(x)
	# on regarde si la sortie est correcte
	acc += torch.argmax(y,1) == torch.argmax(t,1)
# on affiche le pourcentage de bonnes réponses
print(acc/data_test.shape[0])
