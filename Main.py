import neuralnetworks
import settings
import PIL

import sys
from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtCore import QThread, QThreadPool
from PyQt5.QtWidgets import QWidget, QLCDNumber, QSlider, QVBoxLayout, QApplication, QPushButton, QFileDialog, QMessageBox
from MainWindow import Ui_MainWindow
from CreationWindow import Ui_Dialog

import matplotlib.pyplot as plt

import numpy as np
import imageio

import random
import json
import time




class DataThread(QThread):
	def __init__(self, window, parent = None):
		super(DataThread, self).__init__(parent)
		self.window = window

	def run(self):
		name = get_filename(self.window.filename, True)
		if self.window.data_type == "train":
			self.window.ui.training_file.setText("Data loading...")
			with open(self.window.filename, "r") as file:
				self.window.training_data = json.load(file)
			self.window.ui.training_file.setText(name)
		else:
			self.window.ui.testing_file.setText("Data loading...")
			with open(self.window.filename, "r") as file:
				self.window.testing_data = json.load(file)
			self.window.ui.testing_file.setText(name)



class TrainingThread(QThread):
	def __init__(self, window, parent = None):
		super(TrainingThread, self).__init__(parent)
		self.window = window
		self.epoch_count = 0
		self.count = 0
		self.epoch_samples = min(self.window.st.epochsize, len(self.window.training_data))
		self.testing_samples = min(self.window.st.test_samples, len(self.window.testing_data))
		self.total_epoch_samples = self.epoch_samples + self.testing_samples
		self.total_samples = self.total_epoch_samples * self.window.st.epochs
		self.window.upperLimit_epoch.emit(self.total_epoch_samples)
		self.window.upperLimit_training.emit(self.total_samples)
		self.best_epoch = 1
		self.summ = 0
		self.result = 0
		self.best_result = 0
		self.total_time = 0
		self.window.results = []
		self.population = [i for i in range(0, 60000)]

	def run(self):
		start = time.time()
		for j in range(self.window.st.epochs):
			self.epoch_count = 0
			cur_data = random.sample(self.population, self.epoch_samples)
			for i in range(self.epoch_samples):
				sample = self.window.training_data[cur_data[i]]
				self.window.network.training(sample[1:], sample[0])
				self.epoch_count += 1
				self.count += 1
				if self.count % 10 == 0:
					self.window.progressChanged_epoch.emit(self.epoch_count)
					self.window.progressChanged_training.emit(self.count)
					self.window.timeChanged.emit(f"{time.time() - start:.{2}f}")
			self.summ = 0
			for i in range(self.testing_samples):
				sample = self.window.training_data[i]
				self.summ += check_ans(sample[0], self.window.network.query(sample[1:]))
				self.epoch_count += 1
				self.count += 1
				if self.count % 10 == 0:
					self.window.progressChanged_epoch.emit(self.epoch_count)
					self.window.progressChanged_training.emit(self.count)
					self.window.timeChanged.emit(f"{time.time() - start:.{2}f}")
			self.total_time = time.time() - start
			self.window.avgEpochTimeChanged.emit(f"{self.total_time/float(j + 1):.{2}f}")
			self.result = self.summ / self.testing_samples
			self.window.results.append(self.result) 
			if self.best_result < self.result:
				self.best_result = self.result
				self.best_epoch = j + 1
				self.window.best_network = self.window.network
				self.window.bestEpochChanged.emit(str(self.best_epoch))
				self.window.bestResultChanged.emit(str(self.best_result))
			self.window.resultChanged.emit(str(self.result))
			self.window.epochChanged.emit(str(j + 1))
			if j == self.window.st.epochs - 1:
				self.window.final_network = self.window.network
			if self.window.stop == True:
				self.window.stop = False
				break
		self.window.toggle_window()
		

class creationwindow(QtWidgets.QDialog):
	def __init__(self, window, parent = None):
		super(creationwindow, self).__init__(parent)
		self.ui = Ui_Dialog()
		self.ui.setupUi(self)

		self.window = window

		self.ui.inodes.setText(str(window.st.inodes))
		self.ui.onodes.setText(str(window.st.onodes))
		self.ui.hlayers.setText(str(window.st.hlayers))
		self.ui.hnodes.setText(str(window.st.hnodes))
		self.ui.networks_name.setText(window.st.name)
		
		self.ui.networks_name.editingFinished.connect(self.name_setting)
		self.ui.inodes.editingFinished.connect(self.inodes_setting)
		self.ui.onodes.editingFinished.connect(self.onodes_setting)
		self.ui.hlayers.editingFinished.connect(self.hlayers_setting)
		self.ui.hnodes.editingFinished.connect(self.hnodes_setting)
		self.ui.buttonBox.accepted.connect(self.create_network)
		self.ui.buttonBox.rejected.connect(self.cancel)

		self.temp = {"name": window.st.name, "inodes": window.st.inodes, "onodes": window.st.onodes, "hlayers": window.st.hlayers, "hnodes": window.st.hnodes}
		self.setWindowTitle("New Network")
		self.show()

	def name_setting(self):
		self.temp["name"] = self.ui.networks_name.text()

	def inodes_setting(self):
		number = self.ui.inodes.text()
		if number.isdigit():
			self.temp["inodes"] = int(number)
		else:
			self.ui.inodes.setText(str(self.temp["inodes"]))

	def onodes_setting(self):
		number = self.ui.onodes.text()
		if number.isdigit():
			self.temp["onodes"] = int(number)
		else:
			self.ui.onodes.setText(str(self.temp["onodes"]))

	def hlayers_setting(self):
		number = self.ui.hlayers.text()
		if number.isdigit():
			self.temp["hlayers"] = int(number)
		else:
			self.ui.hlayers.setText(str(self.temp["hlayers"]))

	def hnodes_setting(self):
		number = self.ui.hnodes.text()
		if number.isdigit():
			self.temp["hnodes"] = int(number)
		else:
			self.ui.hnodes.setText(str(self.temp["hnodes"]))

	def create_network(self):
		temp = self.temp
		self.window.st.hlayers = self.temp["hlayers"]
		self.window.st.reinit_mllr(True)
		self.temp["learning_rate"] = self.window.st.mllr
		if self.temp["hlayers"] == 1:
			self.window.network = neuralnetworks.SHLNN(temp["inodes"], temp["hnodes"], temp["onodes"], learningrate = self.temp["learning_rate"])
		else:
			self.window.network = neuralnetworks.MHLNN(temp["inodes"], temp["hlayers"], temp["hnodes"], temp["onodes"], learningrate = self.temp["learning_rate"])
		self.window.st.name = temp["name"]
		self.window.networks.append([self.window.st.name, self.window.network])
		self.window.response = "ok"
		self.close()

	def cancel(self):
		self.window.response = "cancel"
		self.close()


class mainwindow(QtWidgets.QMainWindow):
	progressChanged_epoch = QtCore.pyqtSignal(int)
	progressChanged_training = QtCore.pyqtSignal(int)
	upperLimit_epoch = QtCore.pyqtSignal(int)
	lowerLimit_epoch = QtCore.pyqtSignal(int)
	upperLimit_training = QtCore.pyqtSignal(int)
	lowerLimit_training = QtCore.pyqtSignal(int)
	timeChanged = QtCore.pyqtSignal(str)
	avgEpochTimeChanged = QtCore.pyqtSignal(str)
	epochChanged = QtCore.pyqtSignal(str)
	resultChanged = QtCore.pyqtSignal(str)
	bestResultChanged = QtCore.pyqtSignal(str)
	bestEpochChanged = QtCore.pyqtSignal(str)

	def __init__(self):
		super(mainwindow, self).__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.threadpool = QThreadPool()

		self.st = settings.Settings()

		self.ui.lr_value.display(self.st.lr)
		self.ui.learning_rate.setValue(int(self.st.lr * 100000))
		self.ui.epoch_size.setText(str(self.st.epochsize))
		self.ui.epochs.setText(str(self.st.epochs))
		self.ui.backquery.setText(str(self.st.backquery))
		self.ui.samples.setText(str(self.st.test_samples))

		self.ui.create.clicked.connect(self.creating_network)
		self.ui.networks_list.itemClicked.connect(self.selecting_network)
		self.ui.open_network.clicked.connect(self.opening_network)
		self.ui.delete_network.clicked.connect(self.deleting_network)
		self.ui.open_training_data.clicked.connect(self.opening_training_data)
		self.ui.open_testing_data.clicked.connect(self.opening_testing_data)
		self.ui.learning_rate.sliderMoved.connect(self.lr_display)
		self.ui.learning_rate.sliderReleased.connect(self.setting_lr)
		self.ui.learning_rate.sliderPressed.connect(self.lr_display)
		self.ui.different_lr_check.stateChanged.connect(self.toggle_diff_mllr)
		self.ui.min_lr.sliderMoved.connect(self.min_lr_display)
		self.ui.min_lr.sliderPressed.connect(self.min_lr_display)
		self.ui.min_lr.sliderReleased.connect(self.setting_min_lr)
		self.ui.max_lr.sliderMoved.connect(self.max_lr_display)
		self.ui.max_lr.sliderPressed.connect(self.max_lr_display)
		self.ui.max_lr.sliderReleased.connect(self.setting_max_lr)
		self.ui.epoch_size.editingFinished.connect(self.setting_epoch_size)
		self.ui.epochs.editingFinished.connect(self.setting_epochs)
		self.ui.training_backquery.stateChanged.connect(self.toggle_backquery)
		self.ui.backquery.editingFinished.connect(self.setting_backquery_period)
		self.ui.samples.editingFinished.connect(self.setting_samples)
		self.ui.start_training.clicked.connect(self.training)
		self.ui.save_best_network.clicked.connect(self.saving_best)
		self.ui.save_final_network.clicked.connect(self.saving_final)
		self.ui.show_stats.clicked.connect(self.showing_stats)
		self.ui.open_query_file.clicked.connect(self.opening_querying_file)
		self.ui.draw_a_number.clicked.connect(self.drawing_a_number)
		self.ui.stopButton.clicked.connect(self.stop_training)

		self.progressChanged_epoch.connect(self.ui.epoch_progress.setValue)
		self.progressChanged_training.connect(self.ui.training_progress.setValue)
		self.upperLimit_epoch.connect(self.ui.epoch_progress.setMaximum)
		self.upperLimit_training.connect(self.ui.training_progress.setMaximum)
		self.lowerLimit_epoch.connect(self.ui.epoch_progress.setMinimum)
		self.lowerLimit_training.connect(self.ui.training_progress.setMinimum)
		self.timeChanged.connect(self.ui.total_time.setText)
		self.avgEpochTimeChanged.connect(self.ui.average_epoch_time.setText)
		self.epochChanged.connect(self.ui.epoch.setText)
		self.resultChanged.connect(self.ui.result.setText)
		self.bestEpochChanged.connect(self.ui.best_epoch.setText)
		self.bestResultChanged.connect(self.ui.best_result.setText)

		self.backquery_state = False
		self.mllr_state = False
		self.training_data_loaded = False
		self.testing_data_loaded = False
		self.toggle = True
		self.networks = []
		self.network = None
		self.best_network = None
		self.results = None
		self.training_data = []
		self.testing_data = []
		self.response = ""
		self.filename = ""
		self.data_type = ""
		self.stop = False

		self.thread = None
		self.additional_thread = None

	def selecting_network(self):
		row = self.ui.networks_list.currentRow()
		self.network = self.networks[row][1]
		self.st.name = self.networks[row][0]
		self.change_labels(self.network)

	def opening_network(self):
		self.filename = QFileDialog.getOpenFileName(self, 'Open file')[0]
		if self.filename != "":
			self.st.name = get_filename(self.filename)
			self.ui.networks_list.addItem(self.st.name)
			self.ui.networks_list.setCurrentRow(self.ui.networks_list.count() - 1)
			with open(self.filename, "r") as file:
				data = json.load(file)
				if data[0][2] > 1:
					self.network = neuralnetworks.MHLNN()
				else:
					self.network = neuralnetworks.SHLNN()
				self.network.load_network(data, file = False)
				self.networks.append([self.st.name, self.network])
			self.change_labels(self.network)
			if self.ui.networks_list.count() == 1:
				self.ui.delete_network.setEnabled(True)
				self.ui.querying_data_label.setEnabled(True)
				self.ui.querying_number_label.setEnabled(True)
				self.ui.querying_number.setEnabled(True)
				self.ui.open_query_file.setEnabled(True)
				self.toggling_start()

	def deleting_network(self):
		row = self.ui.networks_list.currentRow()
		self.st.name = self.networks[row][0]
		msg = "Are you sure you want to delete \"" + self.st.name + "\"?"
		reply = QMessageBox.question(self, 'Warning!', msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
		if reply == QMessageBox.Yes:
			item = self.ui.networks_list.takeItem(row)
			del(item)
			self.networks.pop(row)
		if self.ui.networks_list.count() == 0:
			self.ui.inodes.setText("")
			self.ui.onodes.setText("")
			self.ui.hnodes.setText("")
			self.ui.hlayers.setText("")
			self.ui.delete_network.setEnabled(False)
			self.ui.start_training.setEnabled(False)
			self.ui.querying_data_label.setEnabled(False)
			self.ui.querying_number_label.setEnabled(False)
			self.ui.open_query_file.setEnabled(False)
			self.ui.querying_number.setEnabled(False)


	def creating_network(self):
		reply = creationwindow(window = self)
		reply.exec()
		if self.response == "ok":
			self.st.name = self.networks[-1][0]
			self.network = self.networks[-1][1]
			self.ui.networks_list.addItem(self.st.name)
			self.ui.networks_list.setCurrentRow(self.ui.networks_list.count() - 1)
			self.change_labels(self.network)
			if self.ui.networks_list.count() == 1:
				self.ui.delete_network.setEnabled(True)
				self.toggling_start()
				self.toggling_query()

	def opening_training_data(self):
		self.filename = QFileDialog.getOpenFileName(self, 'Open file')[0]
		if self.filename != "":
			self.data_type = "train"
			self.thread = DataThread(window = self)
			self.thread.start()
			self.training_data_loaded = True
			self.toggling_start()
			self.toggling_query()

	def opening_testing_data(self):
		self.filename = QFileDialog.getOpenFileName(self, 'Open file')[0]
		if self.filename != "":
			self.data_type = "test"
			self.thread = DataThread(window = self)
			self.thread.start()
			self.testing_data_loaded = True
			self.toggling_start()
			self.toggling_query()


	def change_labels(self, network):
		self.st.inodes = network.inodes
		self.st.onodes = network.onodes
		self.st.hlayers = network.hlayers
		self.st.hnodes = network.hnodes
		self.ui.inodes.setText(str(network.inodes))
		self.ui.onodes.setText(str(network.onodes))
		self.ui.hlayers.setText(str(network.hlayers))
		self.ui.hnodes.setText(str(network.hnodes))
		self.ui.querying_number.setText("")

	def toggling_start(self):
		if (self.testing_data_loaded and self.training_data_loaded):
			if self.ui.networks_list.count() > 0:
				self.ui.start_training.setEnabled(True)

	def toggling_query(self):
		if self.ui.networks_list.count() > 0:
			self.ui.querying_data_label.setEnabled(True)
			self.ui.querying_number_label.setEnabled(True)
			self.ui.open_query_file.setEnabled(True)
			self.ui.querying_number.setEnabled(True)

	def toggle_window(self):
		if self.toggle == True:
			self.ui.Label3.setText("Learning in progress")
		else:
			self.ui.Label3.setText("Training and querying")
		self.toggle = not self.toggle
		self.ui.delete_network.setEnabled(self.toggle)
		self.ui.training_data_label.setEnabled(self.toggle)
		self.ui.training_file.setEnabled(self.toggle)
		self.ui.open_training_data.setEnabled(self.toggle)
		self.ui.testing_data_label.setEnabled(self.toggle)
		self.ui.testing_file.setEnabled(self.toggle)
		self.ui.open_testing_data.setEnabled(self.toggle)
		self.ui.start_training.setEnabled(self.toggle)
		self.ui.save_best_network.setEnabled(self.toggle)
		self.ui.save_final_network.setEnabled(self.toggle)
		self.ui.show_stats.setEnabled(self.toggle)
		self.ui.number_of_epochs_label.setEnabled(self.toggle)
		self.ui.epochs.setEnabled(self.toggle)
		self.ui.networks_list.setEnabled(self.toggle)
		self.ui.create.setEnabled(self.toggle)
		self.ui.open_network.setEnabled(self.toggle)
		self.ui.querying_data_label.setEnabled(self.toggle)
		self.ui.querying_number_label.setEnabled(self.toggle)
		self.ui.open_query_file.setEnabled(self.toggle)
		self.ui.querying_number.setEnabled(self.toggle)

	def stop_training(self):
		self.stop = True
		self.ui.stopButton.setEnabled(False)

	def lr_display(self):
		self.ui.lr_value.display(self.ui.learning_rate.value() / 1000000.0)

	def min_lr_display(self):
		self.ui.lr_value.display(self.ui.min_lr.value() / 1000000.0)

	def max_lr_display(self):
		self.ui.lr_value.display(self.ui.max_lr.value() / 1000000.0)

	def setting_lr(self):
		lr = self.ui.learning_rate.value()
		self.st.lrmin = lr / 1000000
		self.st.reinit_mllr(True)

	def setting_min_lr(self):
		lr = self.ui.min_lr.value()
		if lr > self.ui.max_lr.value():
			self.st.lrmax = lr / 1000000
			self.ui.max_lr.setSliderPosition(lr)
		#self.ui.max_lr.setMinimum(lr)
		self.st.lrmin = lr
		self.st.reinit_mllr()

	def setting_max_lr(self):
		lr = self.ui.max_lr.value()
		if lr < self.ui.min_lr.value():
			self.st.lrmin = lr / 1000000
			self.ui.min_lr.setSliderPosition(lr)
		#self.ui.min_lr.setMaximum(lr)
		self.st.lrmax = lr / 1000000
		self.st.reinit_mllr()
		

	def toggle_diff_mllr(self):
		state = self.mllr_state
		self.ui.min_lr_label.setEnabled(not state)
		self.ui.max_lr_label.setEnabled(not state)
		self.ui.learning_rate.setEnabled(state)
		self.ui.min_lr.setEnabled(not state)
		self.ui.max_lr.setEnabled(not state)
		self.mllr_state = not state
		self.st.reinit_mllr(not state)

	def setting_epochs(self):
		number = self.ui.epochs.text()
		if number.isdigit():
			self.st.epochs = int(number)
		else:
			self.ui.epochs.setText(str(self.st.epochs))
		

	def setting_epoch_size(self):
		number = self.ui.epoch_size.text()
		if number.isdigit():
			self.st.epochsize = int(number)
		else:
			self.ui.epoch_size.setText(str(self.st.epochsize))

	def toggle_backquery(self):
		state = self.backquery_state
		self.ui.backquery_period_label.setEnabled(not state)
		self.ui.backquery.setEnabled(not state)
		self.backquery_state = not state

	def setting_backquery_period(self):
		number = self.ui.backquery.text()
		if number.isdigit():
			self.st.backquery = int(number)
		else:
			self.ui.backquery.setText(str(self.st.backquery))

	def setting_samples(self):
		number = self.ui.samples.text()
		if number.isdigit():
			self.st.testing_samples = int(number)
		else:
			self.ui.samples.setText(str(self.st.test_samples))

	def training(self):
		self.toggle_window()
		self.ui.stopButton.setEnabled(True)
		self.thread = TrainingThread(window = self)
		self.thread.start()

	def saving_best(self):
		fname = QFileDialog.getSaveFileName(self, 'Save file')[0]
		if fname != "":
			self.best_network.save_network(fname)

	def saving_final(self):
		fname = QFileDialog.getSaveFileName(self, 'Save file')[0]
		if fname != "":
			self.final_network.save_network(fname)

	def showing_stats(self):
		self.ui.show_stats.setEnabled(False)
		x = np.arange(1, len(self.results) + 1, 1)
		fig = plt.figure()
		ax1 = plt.subplot(251)
		plt.imshow(self.network.back_query(get_target(0, self.st)).reshape((28, 28)), cmap = self.st.coloring)
		ax2 = plt.subplot(252)
		plt.imshow(self.network.back_query(get_target(1, self.st)).reshape((28, 28)), cmap = self.st.coloring)
		ax3 = plt.subplot(253)
		plt.imshow(self.network.back_query(get_target(2, self.st)).reshape((28, 28)), cmap = self.st.coloring)
		ax4 = plt.subplot(254)
		plt.imshow(self.network.back_query(get_target(3, self.st)).reshape((28, 28)), cmap = self.st.coloring)
		ax5 = plt.subplot(255)
		plt.imshow(self.network.back_query(get_target(4, self.st)).reshape((28, 28)), cmap = self.st.coloring)
		ax6 = plt.subplot(256)
		plt.imshow(self.network.back_query(get_target(5, self.st)).reshape((28, 28)), cmap = self.st.coloring)
		ax7 = plt.subplot(257)
		plt.imshow(self.network.back_query(get_target(6, self.st)).reshape((28, 28)), cmap = self.st.coloring)
		ax8 = plt.subplot(258)
		plt.imshow(self.network.back_query(get_target(7, self.st)).reshape((28, 28)), cmap = self.st.coloring)
		ax9 = plt.subplot(259)
		plt.imshow(self.network.back_query(get_target(8, self.st)).reshape((28, 28)), cmap = self.st.coloring)
		ax10 = plt.subplot(2,5,10)
		plt.imshow(self.network.back_query(get_target(9, self.st)).reshape((28, 28)), cmap = self.st.coloring)
		results_fig = plt.figure()
		results_fig.set_facecolor(self.st.back_color)
		ax = plt.subplot()
		ax.scatter(x, self.results, color = "navy", s = 15)
		ax.plot(x, self.results, color = "navy")
		plt.show()
		self.ui.show_stats.setEnabled(True)

	def opening_querying_file(self):
		fname = QFileDialog.getOpenFileName(self, 'Open file')[0]
		if fname != "":
			img = PIL.Image.open(fname)
			img = img.resize((28, 28), PIL.Image.ANTIALIAS)
			img.save("query.png")
			img_data = imageio.imread("query.png", as_gray = True)
			img_data = 255.0 - img_data.reshape(784)
			img_data = (img_data / 256.0 * 0.99) + 0.01
			query_res = self.network.query(img_data)
			query_res = [float(i[0]) for i in query_res]
			fig = plt.figure()
			fig.set_facecolor("seashell")
			number = plt.subplot(211)
			number.imshow(img_data.reshape((28,28)), cmap = "viridis")
			nodes = plt.subplot(212)
			nodes.bar(np.arange(10), query_res)
			plt.show()
			self.ui.querying_number.setText(str(get_number(query_res)))

	def drawing_a_number(self):
		pass


def get_filename(string, form = False):
	if form == False:
		return string.split("/")[-1].split(".")[0]
	else:
		return string.split("/")[-1]

def check_ans(test_data, NN_data):
	for i in range(10):
		if test_data[i] == max(test_data):
			if NN_data[i] == max(NN_data):
				return 1
			else:
				return 0

def get_data(filename, js = True):
	with open(filename, "r") as file:	
		if js == True:
			data = json.load(file)
			return data
		else:
			lines = file.readlines()
			data = []
			for i in range(min(st.epochsize, len(lines))):
				data.append([int(k) for k in lines[i].strip().split(",")])
				temp = get_target(data[i][0])
				data[-1] = [k / 255.0 for k in data[i]]
				data[-1][0] = temp
			return data

def save_data(data, filename):
	with open(filename, "w") as file:
		json.dump(data, file)

def get_number(query):
	for i in range(10):
		if query[i] == max(query):
			return i

def get_target(number, st):
	target = [st.mismatch for i in range(10)]
	target[number] = st.match
	return target

def train(network, data, epochs):
	for i in range(epochs):
		for sample in data:
			network.training(sample[1:], sample[0])
		print(f"Epoch: {i + 1}\nResult: {test(network, data)}")

def test(network, data):
	summ = 0
	for sample in data:
		summ += check_ans(sample[0], network.query(sample[1:]))
	return summ / len(data)


app = QtWidgets.QApplication([])
application = mainwindow()
application.show()
sys.exit(app.exec())