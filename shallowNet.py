import gzip
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


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

((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
test_dataset = torch.utils.data.TensorDataset(data_test,label_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model = ShallowNet(784, hidden_num, 10)

# on initiliase l'optimiseur
loss_func = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.SGD(model.parameters(), lr=eta)

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
		
	# test du modèle (on évalue la progression pendant l'apprentissage)
	acc = 0.
	# on lit toutes les donnéees de test
	for x,t in test_loader:
		# on calcule la sortie du modèle
		y = model(x)
		# on regarde si la sortie est correcte
		acc += torch.argmax(y,1) == torch.argmax(t,1)
	# on affiche le pourcentage de bonnes réponses
	print(acc/data_test.shape[0])

