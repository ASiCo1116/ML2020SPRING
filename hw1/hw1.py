import sys
import csv
import ipdb
import numpy as np
import argparse
import matplotlib.pyplot as plt

from IPython import embed

def preprocess(array):
	array[np.isnan(array)] = 0.0
	mean = np.mean(array, axis = 0).reshape(1, array.shape[1])
	std = np.std(array, axis = 0).reshape(1, array.shape[1])
	array = np.subtract(array, mean)
	array = np.divide(array, std)
	# array[np.isnan(array)] = 0.0
	return array.astype(np.float64)

def train_test_split(arr1, arr2, ratio):
	test_idx = np.random.choice(arr1.shape[0], size = int(arr1.shape[0] * ratio), replace = False)
	train_idx = np.array([index for index in range(arr1.shape[0]) if index not in test_idx])
	return arr1[train_idx, :], arr1[test_idx, :], arr2[train_idx, :], arr2[test_idx, :]

# def feature(list_):
# 	return [idx for idx in range(18) if idx not in list_]

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

	x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, float(sys.argv[3]))

	w = np.zeros((18 * 9 + 1, 1))
	lr = float(sys.argv[1])
	epochs = int(sys.argv[2])
	train_loss_his = []
	valid_loss_his = []
	best_loss = 100000.0
	adagrad = np.zeros((18 * 9 + 1, 1))
	for epoch in range(epochs):
		train_loss = np.sqrt((np.sum(np.power(y_train - x_train.dot(w), 2)) / x_train.shape[0]))
		valid_loss = np.sqrt((np.sum(np.power(y_valid - x_valid.dot(w), 2)) / x_valid.shape[0]))
		if epoch % 100 == 0:
			if valid_loss < best_loss:
				best_loss = valid_loss
				print('best weight saved!')
				np.save('w.npy', w)
			train_loss_his.append(train_loss)
			valid_loss_his.append(valid_loss)
			print(f'epoch: [{epoch + 1}]/[{epochs}]\ttrain loss: {train_loss:.3f}\tvalid loss: {valid_loss:.3f}')
		gradient = 2 * np.dot(x_train.T, x_train.dot(w) - y_train)
		adagrad += gradient ** 2
		w = w - lr * gradient / (adagrad + 1e-7) ** .5
	plt.figure()
	plt.plot(np.arange(epochs / 100), train_loss_his)
	plt.plot(np.arange(epochs / 100), valid_loss_his)
	plt.show()
	plt.savefig('loss.png')
		

def testing(npy):
	test = np.genfromtxt('./test.csv', delimiter = ',', encoding = 'big5')
	test = test[:, 2:]
	test_split = np.vsplit(test, test.shape[0] / 18)
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