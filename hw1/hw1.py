import sys
import csv
import ipdb
import numpy as np

from IPython import embed

def preprocess(array):
	array[np.isnan(array)] = 0.0
	array = array.astype(np.float64)
	array = np.subtract(array, array.min(axis = 1).reshape(18, 1))
	array = np.divide(array, array.max(axis = 1).reshape(18, 1))
	array[np.isnan(array)] = 0.0
	return array

def training():
	train = np.genfromtxt('./train.csv', delimiter = ',', encoding = 'big5')
	train = train[1:, 3:]
	train_split = np.vsplit(train, train.shape[0] / 18)
	train = np.hstack(train_split).astype(np.float64)
	train = preprocess(train)
	w = np.random.randn(1, 18 * 9).astype(np.float64)
	lr = 1e-4
	epochs = 100
	best_loss = 1000000
	best_weight = np.zeros((1, 18 * 9))
	for epoch in range(epochs):
		epoch_loss = 0.0
		for i in range(train.shape[1] - 9):
			loss = train[9, 9 + i] - w.dot(train[:, i:i+9].reshape(18 * 9, 1))
			epoch_loss += np.sum(loss ** 2)
			w += lr * 2 * loss * train[:, i:i+9].reshape(1, 18 * 9)
			if epoch_loss < best_loss:
				best_loss = epoch_loss
				best_weight = w
				np.save('w.npy', best_weight)
				print('Best weight saved!')
		print(f'epoch: [{epoch + 1}]/[{epochs}]\tepoch loss: {epoch_loss:.3f}')

def testing(npy):
	test = np.genfromtxt('./test.csv', delimiter = ',', encoding = 'big5')
	test = test[:, 2:]
	test_split = np.vsplit(test, test.shape[0] / 18)
	test = np.hstack(test_split)
	test = preprocess(test)
	weight = np.load(npy)
	ans = np.zeros((240, ))
	for i in range(test.shape[0], 18):
		ans[i] = np.dot(weight, test[i:i+18].reshape(18 * 9, 1))
	with open('res.csv', 'w', newline = '') as file:
		csvwriter = csv.writer(file)
		title = ['id', 'value']
		for i in range(240):
			csvwriter.writerow([f'id_{i}', ans[i]])


if __name__ == '__main__':
	training()
	testing('./w.npy')