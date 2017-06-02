#!/usr/bin/env python

'''Interface for entering observations for Cops and Robots POMDP experiments.
'''

__author__ = "Ian Loefgren"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Ian Loefgren"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

import sys
import rospy
import matplotlib
import time

matplotlib.use('Qt5Agg')
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont

from interface_elements import *

from observation_interface.srv import *
from observation_interface.msg import *

class ObservationInterface(QMainWindow):

    def __init__(self):

        self.app_name = 'Cops and Robots 2.0'

        super(QMainWindow,self).__init__()
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.initUI()

    def initUI(self):

        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        # create quit button and add at top left corner
        self.quit_btn = QPushButton('QUIT')
        self.quit_btn.clicked.connect(self.close)
        self.main_layout.addWidget(self.quit_btn,0,0)

        # create and add instances of all elements

        # left side <- includes all video feeds
        self.cop_video = CopVideo()
        self.cam_1 = SecurityCamera(1,'Kitchen')
        self.cam_2 = SecurityCamera(2,'Hallway')
        self.cam_3 = SecurityCamera(3,'Billiards Room')

        self.main_layout.addWidget(self.cop_video,1,1,1,2)
        self.main_layout.addWidget(self.cam_1,2,1)
        self.main_layout.addWidget(self.cam_2,2,2)
        self.main_layout.addWidget(self.cam_3,2,3)
        # self.left_column = QVBoxLayout()
        # self.main_layout.addLayout(self.left_column,0,1)

        # right side <- includes all questions and belief map
        self.robot_pull = RobotPull()
        self.human_push = HumanPush()
        self.belief_map = MapDisplay()

        self.main_layout.addWidget(self.robot_pull,1,3)
        self.main_layout.addWidget(self.belief_map,1,4,1,2)
        self.main_layout.addWidget(self.human_push,2,4,1,2)
        # self.right_column = QVBoxLayout()
        # self.right_column.addWidget(self.robot_pull)
        # self.main_layout.addLayout(self.right_column,0,2)


        # self.main_layout.setRowMinimumWidth()

        # self.main_layout.addWidget(self.robot_pull,1,1)

        self.robot_pull.update([0,2,1])

        self.main_layout.setColumnMinimumWidth(0,100)
        # self.main_layout.setColumnMinimumWidth(1,800)
        # self.main_layout.setColumnMinimumWidth(4,300)
        # self.main_layout.setColumnMinimumWidth(5,1000)
        # self.main_layout.setRowMinimumHeight(1,600)
        # self.main_layout.setRowMinimumHeight(2,600)

        self.setWindowTitle(self.app_name)
        self.showMaximized()
        # self.robot_pull.show()
        # self.belief_map.show()

    def quit(self):

        dialog_reply = QMessageBox.warning(self,'Quit', \
                    'Are you sure you want to quit?', \
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if dialog_reply == QMessageBox.Yes:
            self.close()

    def closeEvent(self,event):
        self.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    obs_app = ObservationInterface()
    sys.exit(app.exec_())
