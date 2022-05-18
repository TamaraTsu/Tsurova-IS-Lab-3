from network import Neural_Network
from functions import *
import numpy as np
import matplotlib.pyplot as plt

def main():
	n = Neural_Network((10, 10), 0.02, 3, 20)

	epochs = 20

	loss_epochs = []
	accuracy_epochs = []

	for epoch in range(epochs):
		print(f"Training on epoch-{epoch + 1}")
		n.train_network(epoch)
		accuracy, loss = n.get_average_train_statistics()
		loss_epochs.append(loss) 
		accuracy_epochs.append(accuracy) 

	n.save_training_params()

	visualize_growth(accuracy=accuracy_epochs, loss=loss_epochs, x_param=[(epoch + 1) for epoch in range(epochs)], file_name= "Epochs_Growth.png", isShowing=True)

	n.test_network()

if __name__ == '__main__':
	main()