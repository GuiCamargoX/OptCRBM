# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cl.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!



import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class Ui_Dialog(QtWidgets.QMainWindow):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(659, 582)
        Dialog.setModal(False)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.widget = QtWidgets.QWidget(self.frame)
        self.widget.setEnabled(True)
        self.widget.setObjectName("widget")
        
        layout = QtWidgets.QVBoxLayout(self.widget)
        static_canvas = FigureCanvas(Figure(figsize=(10, 3)))
        self.widget.setFixedHeight(400)
        layout.addWidget(static_canvas)
        layout.setMenuBar( NavigationToolbar(static_canvas, self) )
        self._static_ax = static_canvas.figure.subplots()
        
        self.gridLayout_2.addWidget(self.widget, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 1, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.frame)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_2 = QtWidgets.QFrame(self.groupBox)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_7 = QtWidgets.QLabel(self.frame_2)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 5, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.frame_2)
        self.label_5.setObjectName("label_5")
        self.gridLayout_4.addWidget(self.label_5, 3, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setObjectName("label_6")
        self.gridLayout_4.addWidget(self.label_6, 4, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.frame_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 0, 0, 1, 1)
        self.r_groups = QtWidgets.QLabel(self.frame_2)
        self.r_groups.setText("")
        self.r_groups.setObjectName("r_groups")
        self.gridLayout_4.addWidget(self.r_groups, 0, 1, 1, 1)
        self.r_filter = QtWidgets.QLabel(self.frame_2)
        self.r_filter.setText("")
        self.r_filter.setObjectName("r_filter")
        self.gridLayout_4.addWidget(self.r_filter, 1, 1, 1, 1)
        self.r_cdsteps = QtWidgets.QLabel(self.frame_2)
        self.r_cdsteps.setText("")
        self.r_cdsteps.setObjectName("r_cdsteps")
        self.gridLayout_4.addWidget(self.r_cdsteps, 2, 1, 1, 1)
        self.r_training_error = QtWidgets.QLabel(self.frame_2)
        self.r_training_error.setText("")
        self.r_training_error.setObjectName("r_training_error")
        self.gridLayout_4.addWidget(self.r_training_error, 3, 1, 1, 1)
        self.r_valid_error = QtWidgets.QLabel(self.frame_2)
        self.r_valid_error.setText("")
        self.r_valid_error.setObjectName("r_valid_error")
        self.gridLayout_4.addWidget(self.r_valid_error, 4, 1, 1, 1)
        self.r_test_error = QtWidgets.QLabel(self.frame_2)
        self.r_test_error.setText("")
        self.r_test_error.setObjectName("r_test_error")
        self.gridLayout_4.addWidget(self.r_test_error, 5, 1, 1, 1)
        self.gridLayout_3.addWidget(self.frame_2, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        
        #self.r_test_error.setText('ola')
        #self.set_test_error_result('e')

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", " Generation Curve "))
        self.groupBox.setTitle(_translate("Dialog", "Best Hyperparameter"))
        self.label_7.setText(_translate("Dialog", "Test error:"))
        self.label_5.setText(_translate("Dialog", "Training error :"))
        self.label_3.setText(_translate("Dialog", "Filter size:"))
        self.label_6.setText(_translate("Dialog", "Validation error:"))
        self.label_4.setText(_translate("Dialog", "CD steps:"))
        self.label_2.setText(_translate("Dialog", "Groups:"))

    def set_groups_result(self, number_groups):
        self.r_groups.setText(number_groups )
    
    def set_filtersize_result(self,number_filter):
        self.r_filter.setText(number_filter)
        
    def set_cdsteps_result(self,number_cd):
        self.r_cdsteps.setText(number_cd)
        
    def set_training_error_result(self,error):
        self.r_training_error.setText(error)
        
    def set_valid_error_result(self,error):
        self.r_valid_error.setText(error)
        
    def set_test_error_result(self,error):
        self.r_test_error.setText(error)
        
    def set_curve_learning(self, df):
        self._static_ax.plot(df['avg'] , label='Average')
        self._static_ax.plot(df['min'], label='Minimum')
        self._static_ax.set_title('Error X Generation')
        self._static_ax.set_xlabel('Generation')
        self._static_ax.set_ylabel('Root Mean Square Error (RMSE)')
        self._static_ax.legend(loc='upper right')        
    
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

