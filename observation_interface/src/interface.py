#!/usr/bin/env python

'''
Interface for entering observations for Cops and Robots experiments.

Contains the following widgets in a grid layout (the widget code can be found in
interface_elements.py):
    - 3x security camera video feeds receving data from ROS topics
    - 1x cop robot camera feed receiving data from a ROS topic
    - yes/no question panel
    - codebook to assemble human observation statements
    - belief map image display, loaded from directory
'''

__author__ = "Ian Loefgren"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Ian Loefgren"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

import sys
import rospy
import matplotlib
import time
import os

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

    caught_signal = pyqtSignal(str)

    def __init__(self):

        rospy.init_node('obs_interface')
        self.app_name = 'Cops and Robots 2.0'

        super(QMainWindow,self).__init__()
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.initUI()

        rospy.Subscriber('/caught',Caught,self.caught_callback)
        self.caught_pub = rospy.Publisher('/caught_confirm',Caught,queue_size=10)

        self.caught_signal.connect(self.caught_event)

        print('Observation Interface ready.')

    def initUI(self):

        # create the main layout
        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        # create title
        self.title = QLabel(self.app_name)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet(title_style)
        self.main_layout.addWidget(self.title,0,4,1,4,)

        # COHRINT logo
        self.logo = QLabel()
        self.logo_image = QPixmap()
        check = self.logo_image.load(os.path.abspath(os.path.dirname(__file__) + '/black_cohrint_symbshort.png'))
        self.logo_image = self.logo_image.scaled(93,100,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.logo.setPixmap(self.logo_image)
        self.main_layout.addWidget(self.logo,0,12,1,1,Qt.AlignRight)

        # create quit button and add at top left corner
        self.quit_btn = QPushButton('QUIT')
        self.quit_btn.clicked.connect(self.close)
        self.main_layout.addWidget(self.quit_btn,0,0)

        # create and add instances of all elements

        # left side <- includes all video feeds
        cop_name = rospy.get_param("~cop_name")
        self.cop_video = CopVideo(cop_name)
        self.cam_1 = SecurityCamera(1,'Study')
        self.cam_2 = SecurityCamera(2,'Hallway')
        self.cam_3 = SecurityCamera(3,'Kitchen')

        self.main_layout.addWidget(self.cop_video,1,3,4,2,Qt.AlignCenter)
        self.main_layout.addWidget(self.cam_1,1,0,2,2,Qt.AlignCenter)
        self.main_layout.addWidget(self.cam_2,3,0,2,2,Qt.AlignCenter)
        self.main_layout.addWidget(self.cam_3,5,0,2,2,Qt.AlignCenter)

        # right side -> includes all questions and belief map
        self.robot_pull = RobotPull()
        self.human_push = HumanPush()
        self.belief_map = MapDisplay()

        self.main_layout.addWidget(self.robot_pull,5,3,2,3,Qt.AlignTop)
        self.main_layout.addWidget(self.belief_map,1,6,4,6,Qt.AlignCenter)
        self.main_layout.addWidget(self.human_push,5,7,2,5,Qt.AlignTop)

        self.setWindowTitle(self.app_name)
        self.showMaximized()

    def closeEvent(self,event):
        """
        Overidden closeEvent which always displays a confirmation box when the
        user atttempts to close are quit the interface
        """
        dialog_reply = QMessageBox.warning(self,'Quit', \
                    'Are you sure you want to quit?', \
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if dialog_reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def caught_callback(self,msg):
        name = msg.robber
        self.caught_signal.emit(name)

    @pyqtSlot(str)
    def caught_event(self,name):
        name = name.title()
        dialog_reply = QMessageBox.information(self,'Caught?', \
                        'Was '+name+' caught?', \
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        caught = None
        if dialog_reply == QMessageBox.Yes:
            caught = True
        else:
            caught = False

        caught_msg = Caught(name.lower(),caught)
        self.caught_pub.publish(caught_msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    obs_app = ObservationInterface()
    sys.exit(app.exec_())
