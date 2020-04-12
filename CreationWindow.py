# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CreationWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(307, 268)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.networks_name_label = QtWidgets.QLabel(Dialog)
        self.networks_name_label.setObjectName("networks_name_label")
        self.verticalLayout_3.addWidget(self.networks_name_label)
        self.networks_name = QtWidgets.QLineEdit(Dialog)
        self.networks_name.setObjectName("networks_name")
        self.verticalLayout_3.addWidget(self.networks_name)
        self.inodes_label = QtWidgets.QLabel(Dialog)
        self.inodes_label.setObjectName("inodes_label")
        self.verticalLayout_3.addWidget(self.inodes_label)
        self.inodes = QtWidgets.QLineEdit(Dialog)
        self.inodes.setObjectName("inodes")
        self.verticalLayout_3.addWidget(self.inodes)
        self.hlayers_label = QtWidgets.QLabel(Dialog)
        self.hlayers_label.setObjectName("hlayers_label")
        self.verticalLayout_3.addWidget(self.hlayers_label)
        self.hlayers = QtWidgets.QLineEdit(Dialog)
        self.hlayers.setObjectName("hlayers")
        self.verticalLayout_3.addWidget(self.hlayers)
        self.hnodes_label = QtWidgets.QLabel(Dialog)
        self.hnodes_label.setObjectName("hnodes_label")
        self.verticalLayout_3.addWidget(self.hnodes_label)
        self.hnodes = QtWidgets.QLineEdit(Dialog)
        self.hnodes.setObjectName("hnodes")
        self.verticalLayout_3.addWidget(self.hnodes)
        self.onodes_label = QtWidgets.QLabel(Dialog)
        self.onodes_label.setObjectName("onodes_label")
        self.verticalLayout_3.addWidget(self.onodes_label)
        self.onodes = QtWidgets.QLineEdit(Dialog)
        self.onodes.setObjectName("onodes")
        self.verticalLayout_3.addWidget(self.onodes)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.networks_name_label.setText(_translate("Dialog", "Network\'s name:"))
        self.networks_name.setText(_translate("Dialog", "Eve"))
        self.inodes_label.setText(_translate("Dialog", "Number of input neurons:"))
        self.inodes.setText(_translate("Dialog", "784"))
        self.hlayers_label.setText(_translate("Dialog", "Number of hidden layers:"))
        self.hlayers.setText(_translate("Dialog", "1"))
        self.hnodes_label.setText(_translate("Dialog", "Number of neurons in hidden layers:"))
        self.hnodes.setText(_translate("Dialog", "100"))
        self.onodes_label.setText(_translate("Dialog", "Number of output neurons:"))
        self.onodes.setText(_translate("Dialog", "10"))
