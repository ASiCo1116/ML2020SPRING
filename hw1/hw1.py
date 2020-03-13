import sys
import csv
import ipdb
import numpy as np
import argparse

from IPython import embed

def preprocess(array):
	array[np.isnan(array)] = 0.0
	mean = np.mean(array, axis = 0).reshape(1, array.shape[1])
	std = np.std(array, axis = 0).reshape(1, array.shape[1])
	array = np.subtract(array, mean)
	array = np.divide(array, std)
	# array[np.isnan(array)] = 0.0
	return array.astype(np.float64)

def training():
	train = np.genfromtxt('./train.csv', delimiter = ',', encoding = 'big5')
	train = train[1:, 3:]
	train_split = np.vsplit(train, train.shape[0] / 18)
	train = np.hstack(train_split)
	train_x = np.zeros((12 * 471, 18 * 9))
	train_y = np.zeros((12 * 471, 1))
	i = 0
	j = 0
	count = 0
	while i < train.shape[1]:
		train_x[j, :] = train[:, i:i+9].reshape(1, 18 * 9)
		train_y[j][0] = train[9][9 + i]
		if (i + 10) % 480 == 0:
			i += 10
		else:
			i += 1
		j += 1
	train_x = preprocess(train_x)
	b = np.ones((12 * 471, 1))
	train_x = np.concatenate((b, train_x), axis = 1).astype(float)
	w = np.zeros((18 * 9 + 1, 1))
	lr = 1e2
	epochs = 1000
	# best_loss = 10000
	adagrad = np.zeros((18 * 9 + 1, 1))
	for epoch in range(epochs):
		loss = np.sqrt((np.sum(np.power(train_y - train_x.dot(w), 2)) / train_x.shape[0]))
		if epoch % 100 == 0:
			print(f'epoch: [{epoch + 1}]/[{epochs}]\tepoch loss: {loss:.3f}')
		gradient = 2 * np.dot(train_x.T, train_x.dot(w) - train_y)
		adagrad += gradient ** 2
		w = w - lr * gradient / (adagrad + 1e-7) ** .5
		# if loss < best_loss:
		# 	best_loss = loss
	np.save('w.npy', w)
		

def testing(npy):
	test = np.genfromtxt('./test.csv', delimiter = ',', encoding = 'big5')
	test = test[:, 2:]
	test_split = np.vsplit(test, test.shape[0] / 18)
	# test = np.hstack(test_split)
	test = np.zeros((240, 18 * 9))
	for i, split in enumerate(test_split):
		test[i, :] = split.reshape(1, 18 * 9)
	test = preprocess(test)
	b = np.ones((240, 1))
	test = np.concatenate((b, test), axis = 1).astype(float)
	weight = np.load(npy)
	ans = np.zeros((240, 1))
	ans = test.dot(weight)
	with open('res.csv', 'w', newline = '') as file:
		csvwriter = csv.writer(file)
		title = ['id', 'value']
		csvwriter.writerow(title)
		for i in range(240):
			csvwriter.writerow([f'id_{i}', ans[i][0]])


if __name__ == '__main__':
	training()
	testing('./w.npy')