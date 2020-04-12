import numpy as np

class Settings:
	def __init__(self):
		#basic settings
		self.name = "Eve"
		self.inodes = 784
		self.onodes = 10
		self.hnodes = 100
		#for both
		self.lr = 0.08
		#for multi-layer
		self.hlayers = 1
		self.maxlr = 0.08
		self.minlr = 0.08
		self.mllr = np.zeros(self.hlayers + 1)
		for i in range(self.hlayers + 1):
			self.mllr[i] = (self.maxlr - self.minlr)*np.random.random_sample() + self.minlr
		#training
		self.epochs = 5
		self.epochsize = 5000
		self.match = 0.999
		self.mismatch = 0.001
		#querying
		self.test_samples = 2000
		#bitmap
		self.coloring = "viridis"
		#results graph
		self.back_color = "seashell"
		#other
		self.train_json = "Data/train.json"
		self.test_json = "Data/test.json"
		self.train_csv = "Data/train.csv"
		self.test_csv = "Data/test.csv"
		self.network = "Data/network.json"
		self.current_network = ""
		self.backquery = 20

	def reinit_mllr(self, united = False):
		if united:
			self.mllr = [self.lr for i in range(self.hlayers + 1)]
		else:
			for i in range(self.hlayers + 1):
				self.mllr[i] = (self.maxlr - self.minlr)*np.random.random_sample() + self.minlr
