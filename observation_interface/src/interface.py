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

title_style = "\
                    QLabel {    \
                        font-family: Helvetica Neue;    \
                        font-size: 25pt;    \
                        font-weight: 100; \
                        text-align: center;    \
                    }"

logo_style = "\
                    QLabel {    \
                        padding: 0px;   \
                        margin: 0px;    \
                    }"

main_widget_style = "\
                        QWidget {   \
                            background-color: lightgray;    \
                        }"

class ObservationInterface(QMainWindow):

    def __init__(self):

        self.app_name = 'Cops and Robots 1.5'

        super(QMainWindow,self).__init__()
        self.main_widget = QWidget()
        # self.main_widget.setStyleSheet(main_widget_style)
        self.setCentralWidget(self.main_widget)
        self.initUI()

        rospy.init_node('obs_interface')
        print('Observation Interface ready.')

    def initUI(self):

        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)
        # self.main_layout.setAlignment(Qt.AlignTop)

        # create title
        self.title = QLabel(self.app_name)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet(title_style)
        self.main_layout.addWidget(self.title,0,4,1,4,)

        # COHRINT logo
        self.logo = QLabel()
        self.logo_image = QPixmap()
        check = self.logo_image.load('/home/ian/catkin_ws/src/cops-and-robots-2.0/observation_interface/src/black_cohrint_symbshort.png')
        self.logo_image = self.logo_image.scaled(93,100,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.logo.setPixmap(self.logo_image)
        # self.logo.setScaledContents(True)
        self.main_layout.addWidget(self.logo,0,12,1,1,Qt.AlignRight)

        # create quit button and add at top left corner
        self.quit_btn = QPushButton('QUIT')
        self.quit_btn.clicked.connect(self.close)
        self.main_layout.addWidget(self.quit_btn,0,0)

        # create and add instances of all elements

        # left side <- includes all video feeds
        self.cop_video = CopVideo('pris')
        self.cam_1 = SecurityCamera(1,'Study')
        self.cam_2 = SecurityCamera(2,'Hallway')
        self.cam_3 = SecurityCamera(3,'Kitchen')

        self.main_layout.addWidget(self.cop_video,1,3,4,2,Qt.AlignCenter)
        self.main_layout.addWidget(self.cam_1,1,0,2,2,Qt.AlignCenter) #prev: 4 0 2 2
        self.main_layout.addWidget(self.cam_2,3,0,2,2,Qt.AlignCenter) #prev: 4 2 2 2
        self.main_layout.addWidget(self.cam_3,5,0,2,2,Qt.AlignCenter) #prev: 4 4 2 2
        # self.left_column = QVBoxLayout()
        # self.main_layout.addLayout(self.left_column,0,1)

        # right side -> includes all questions and belief map
        self.robot_pull = RobotPull()
        self.human_push = HumanPush()
        self.belief_map = MapDisplay()

        self.main_layout.addWidget(self.robot_pull,5,3,2,3,Qt.AlignTop)
        self.main_layout.addWidget(self.belief_map,1,6,4,6,Qt.AlignCenter)
        self.main_layout.addWidget(self.human_push,5,7,2,5,Qt.AlignTop)
        # self.right_column = QVBoxLayout()
        # self.right_column.addWidget(self.robot_pull)
        # self.main_layout.addLayout(self.right_column,0,2)

        self.setWindowTitle(self.app_name)
        self.showMaximized()

    def closeEvent(self,event):
        dialog_reply = QMessageBox.warning(self,'Quit', \
                    'Are you sure you want to quit?', \
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if dialog_reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    obs_app = ObservationInterface()
    sys.exit(app.exec_())
