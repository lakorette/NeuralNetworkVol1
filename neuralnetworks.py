import numpy as np
import scipy.special as sp
import settings as sett

import json

st = sett.Settings()

def scale(outputs):
		output = outputs
		output -= np.min(output)
		output /= np.max(output)
		output = (output * (st.match - st.mismatch)) + st.mismatch
		return output

#Single Hidden Layer Neural Network class
class SHLNN:
	def __init__(self, inputnodes = st.inodes, hiddennodes = st.hnodes, outputnodes = st.onodes, learningrate = st.mllr):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.hlayers = 1
		self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		self.lr = learningrate
		self.activation = lambda x: sp.expit(x)
		self.deactivation = lambda x: sp.logit(x)

	def training(self, inputs_list, targets_list):
		inputs = np.array(inputs_list, ndmin = 2).T
		targets = np.array(targets_list, ndmin = 2).T
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation(hidden_inputs)
		ll_inputs = np.dot(self.who, hidden_outputs)
		ll_outputs = self.activation(ll_inputs)
		output_errors = targets - ll_outputs
		hidden_errors = np.dot(self.who.T, output_errors)
		self.who += self.lr[1] * np.dot((output_errors * ll_outputs * (1.0 - ll_outputs)), hidden_outputs.T) 
		self.wih += self.lr[0] * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)


	def query(self, inputs_list):
		inputs = np.array(inputs_list, ndmin = 2).T
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation(hidden_inputs) 
		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation(final_inputs)
		return final_outputs

	def back_query(self, outputs_list):
		outputs = np.array(outputs_list, ndmin = 2).T
		final_inputs = self.deactivation(outputs)
		hidden_outputs = np.dot(self.who.T, final_inputs)
		hidden_outputs -= np.min(hidden_outputs)
		hidden_outputs /= np.max(hidden_outputs)
		hidden_outputs *= st.match - st.mismatch
		hidden_outputs += st.mismatch
		hidden_inputs = self.deactivation(hidden_outputs)
		inputs = np.dot(self.wih.T, hidden_inputs)
		return inputs

	def save_network(self, filename = st.network):
		nodes = [self.inodes, self.onodes, 1, self.hnodes]
		layers = []
		layer = []
		for i in range(self.hnodes):
			for j in range(self.inodes):
				layer.append(self.wih[i][j])
		layers.append(layer)
		layer = []
		for i in range(self.onodes):
			for j in range(self.hnodes):
				layer.append(self.who[i][j])
		layers.append(layer)
		network = [nodes, layers]
		with open(filename, "w") as file:
			json.dump(network, file)

	def load_network(self, filename = st.network, file = True):
		if file:
			with open(filename, "r") as file:
				network = json.load(file)
			nodes = network[0]
			self.wih, self.who = [np.array(i) for i in network[1]]
		else:
			nodes = filename[0]
			self.wih, self.who = [np.array(i) for i in filename[1]]
		self.inodes, self.onodes, temp, self.hnodes = [i for i in nodes]
		self.wih = self.wih.reshape((self.hnodes, self.inodes))
		self.who = self.who.reshape((self.onodes, self.hnodes))



#Multi Hidden Layer Neural Network 
class MHLNN:
	def __init__(self, inputnodes = st.inodes, hiddenlayers = st.hlayers, hiddennodes = st.hnodes, outputnodes = st.onodes, learningrate = st.mllr):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.hlayers = hiddenlayers
		self.inputw = np.array(np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
		self.outputw = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		self.hiddenw = np.zeros((self.hlayers - 1, self.hnodes, self.hnodes))
		for i in range(self.hlayers - 1):
			self.hiddenw[i] = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.hnodes)) 
		self.activation = lambda x: sp.expit(x)
		self.deactivation = lambda x: sp.logit(x)
		self.lr = learningrate
	
	def training(self, inputs_list, targets_list):
		inputs = np.array(inputs_list, ndmin = 2).T
		targets = np.array(targets_list, ndmin = 2).T
		hidden_inputs = np.dot(self.inputw, inputs)
		hiddenh_inputs = np.zeros((self.hlayers - 1, self.hnodes, 1))
		hiddenh_outputs = np.zeros((self.hlayers - 1, self.hnodes, 1))
		for i in range(self.hlayers - 1):
			if i == 0:
				hiddenh_outputs[0] = self.activation(hidden_inputs)
				hiddenh_inputs[0] = np.dot(self.hiddenw[0], hiddenh_outputs[0])
			else:
				hiddenh_outputs[i] = self.activation(hiddenh_inputs[i - 1])
				hiddenh_inputs[i] = np.dot(self.hiddenw[i], hiddenh_outputs[i])
		hidden_outputs = self.activation(hiddenh_inputs[-1])
		final_inputs = np.dot(self.outputw, hidden_outputs)
		final_outputs = self.activation(final_inputs)
		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.outputw.T, output_errors)
		#hidden_errors = scale(hidden_errors)
		self.outputw += self.lr[-1] * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
		for i in range(self.hlayers - 2, -1, -1):
			prev_errors = hidden_errors
			if i == self.hlayers - 2:
				self.hiddenw[i] += self.lr[i + 1] * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), hiddenh_outputs[i].T)
			else:
				hidden_errors = np.dot(self.hiddenw[1 + i].T, prev_errors)
				#hidden_errors = scale(hidden_errors)
				self.hiddenw[i] += self.lr[i + 1] * np.dot(hidden_errors * hiddenh_outputs[i + 1] * (1.0 - hiddenh_outputs[i + 1]), hiddenh_outputs[i].T)
		prev_errors = hidden_errors
		hidden_errors = np.dot(self.hiddenw[0].T, prev_errors)
		#hidden_errors = scale(hidden_errors)
		self.inputw += self.lr[0] * np.dot((hidden_errors * hiddenh_outputs[0] * (1.0 - hiddenh_outputs[0])), inputs.T) 

	def query(self, inputs_list):
		inputs = np.array(inputs_list, ndmin = 2).T
		outputs = np.dot(self.inputw, inputs)
		for i in range(self.hlayers - 1):
			inputs = self.activation(outputs)
			outputs = np.dot(self.hiddenw[i], inputs)
		inputs = self.activation(outputs)
		outputs = np.dot(self.outputw, inputs)
		inputs = self.activation(outputs)
		return inputs

	def back_query(self, outputs_list):
		outputs = np.array(outputs_list, ndmin = 2).T
		inputs = self.deactivation(outputs)
		outputs = np.dot(self.outputw.T, inputs)
		outputs = scale(outputs)
		for i in range(self.hlayers - 2, -1, -1):
			inputs = self.deactivation(outputs)
			outputs = np.dot(self.hiddenw[i].T, inputs)
			outputs = scale(outputs)
		inputs = self.deactivation(outputs)
		initial_inputs = np.dot(self.inputw.T, inputs)
		initial_inputs = scale(initial_inputs)
		return initial_inputs

	def save_network(self, filename = st.network):
		nodes = [self.inodes, self.onodes, self.hlayers, self.hnodes]
		layers = []
		layer = []
		for i in range(self.hnodes):
			for j in range(self.inodes):
				layer.append(self.inputw[i][j])
		layers.append(layer)
		for i in range(self.hlayers - 1):
			layer = []
			for j in range(self.hnodes):
				for k in range(self.hnodes):
					layer.append(self.hiddenw[i][j][k])
			layers.append(layer)
		layer = []
		for i in range(self.onodes):
			for j in range(self.hnodes):
				layer.append(self.outputw[i][j])
		layers.append(layer)
		network = [nodes, layers]
		with open(filename, "w") as file:
			json.dump(network, file)

	def load_network(self, filename = st.network, file = True):
		if file:
			with open(filename, "r") as file:
				network = json.load(file)
			nodes = network[0]
			layers = network[1]
		else:
			nodes = filename[0]
			layers = filename[1]
		self.inodes, self.onodes, self.hlayers, self.hnodes = [i for i in nodes]
		self.inputw = np.array(layers[0]).reshape((self.hnodes, self.inodes))
		self.outputw = np.array(layers[-1]).reshape((self.onodes, self.hnodes))
		self.hiddenw = np.zeros((self.hlayers - 1, self.hnodes, self.hnodes))
		for i in range(1, self.hlayers):
			self.hiddenw[i - 1] = np.array(layers[i]).reshape((self.hnodes, self.hnodes))

	def change_lr(self, index, newlr):
		self.lr[index] = newlr