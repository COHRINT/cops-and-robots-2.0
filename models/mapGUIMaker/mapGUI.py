#!/usr/bin/env python

""" Qt Driver for the Map Maker GUI """

__author__ = "LT"
__copyright__ = "Copyright 2017, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "LT"
__email__ = "luba6098@colorado.edu"
__status__ = "Development"

import sys
from PyQt4.QtGui import *

a = QApplication(sys.argv)
# w = QWidget()
# QMessageBox.warning(w,"Message", "Are you sure you want to continue?")

w = QMainWindow()
w.setWindowTitle("Hello World!")
w.resize(400, 400)

mainMenu = w.menuBar()
mainMenu.setNativeMenuBar(True)
fileMenu = mainMenu.addMenu('BIGGY')
fileMenu5 = mainMenu.addMenu('BIG BIGGY')
fileMenu6 = mainMenu.addMenu('BIG BIGGGY')

# mainMenu2 = w.menuBar()
# mainMenu2.setNativeMenuBar(True)
# fileMenu2 = mainMenu2.addMenu('BIGGGGGGGGGY')
#
# mainMenu3 = w.menuBar()
# mainMenu3.setNativeMenuBar(True)
# fileMenu3 = mainMenu3.addMenu('BIGGY3')

exitButton = QAction(QIcon('exit24.png'), 'Exit', w)
exitButton.setShortcut('Ctrl+Q')
exitButton.setStatusTip('Exit application')
exitButton.triggered.connect(w.close)
fileMenu.addAction(exitButton)
# fileMenu2.addAction(exitButton)
# fileMenu3.addAction(exitButton)




# result = QMessageBox.question(w, 'Message', "Do you like Python?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
#
# if result == QMessageBox.Yes:
#     print("Yes.")
# else:
#     print("No.")

w.show()

sys.exit(a.exec_())
