#! /usr/bin/env python
""" GUI that generate maps (.yaml or .json)
For development $ source mapGUI/bin/activate
"""

__author__ = "LT"
__copyright__ = "Copyright 2017, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "LT"
__email__ = "luba6098@colorado.edu"
__status__ = "Development"

import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import pyqtSlot
# from mapGUI import *

class MapGUIMaker(object):

	def __init__(self):
		print("Entering")
		self.Qapp = QApplication(sys.argv)

		w = QWidget()
		w.resize(700, 800)
		w.setWindowTitle("Hello World")

		# Create the actions
		@pyqtSlot()
		def on_click():
		    print('clicked')

		@pyqtSlot()
		def on_press():
		    print('pressed')

		@pyqtSlot()
		def on_release():
		    print('released')

			# Add a button
		btn = QPushButton('Hello World!', w)
		btn.setToolTip('Click to quit!')
		btn.resize(btn.sizeHint())
		btn.move(100, 700)
		btn.clicked.connect(on_click)
		btn.pressed.connect(on_press)
		btn.released.connect(on_release)




		w.show()
		sys.exit(self.Qapp.exec_())



def testGUIDisplay():
	m = MapGUIMaker()

if __name__ == '__main__':
	testGUIDisplay()
