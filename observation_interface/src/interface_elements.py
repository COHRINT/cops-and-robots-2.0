#!/usr/bin/env python

from __future__ import division

"""
This file contains the class definitions of various elements used in the human
interface for the Cops and Robots 2.0 experiment.
"""

__author__ = "Ian Loefgren"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Ian Loefgren"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

import sys
import yaml
import rospy
import struct
import array
import time

import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize, QByteArray, QRect
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QColor
from PyQt5.QtMultimedia import *

from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from observation_interface.msg import *

questions = ["What is your favorite color?",
            "What is your quest?",
            "What is the airspeed of an unladen swallow?",
            "Is this a filler question?",
            "What about this?"]

PullQuestion_style = "\
                        .QWidget {   \
                            min-width: 300px;   \
                            max-width: 300px;   \
                        }"

groupbox_style = "\
                    QGroupBox {  \
                        font-size: 12pt;    \
                        font-weight: 10;    \
                        text-align: center; \
                    }"

yes_btn_style = "\
                QPushButton {   \
                    color: white;   \
                    background-color: green;    \
                    font-size: 8pt; \
                    margin: 5px;    \
                    padding: 5px;   \
                    min-width: 50px; \
                }"

no_btn_style = "\
                QPushButton {   \
                    color: white;   \
                    background-color: darkred;  \
                    font-size: 8pt; \
                    margin: 5px;    \
                    padding: 5px;   \
                    min-width: 50px; \
                }"

null_btn_style = "\
                    QPushButton {   \
                        color: black;   \
                        background-color: skyblue;  \
                        font-size: 8pt; \
                        margin: 5px;    \
                        padding: 5px;   \
                        min-width: 50px; \
                    }"

question_text_style = "\
                        QLabel {    \
                            font: bold; \
                            font-size: 9pt;   \
                        }"

send_btn_style = "\
                    QPushButton {   \
                        color: white;   \
                        background-color: green;    \
                        font-size: 12pt;    \
                        min-height: 35px;   \
                    }"

clear_btn_style = "\
                    QPushButton {   \
                        color: black;   \
                        background-color: lightgray;    \
                        font-size: 11pt;    \
                        min-height: 25px;   \
                    }"

widget_title_style = "\
                QLabel {    \
                    font-size: 12pt;    \
                    font-weight: 10;    \
                    text-align: center; \
                }"

answer_indicator_style = "\
                            .QLabel {   \
                                max-width: 10px;    \
                                max-height: 20px;   \
                            }"


class RobotPull(QWidget):
    """
    The robot pull questions widget. Displays questions chosen by robot and their
    associated value of information, and presents user with 'yes' and 'no' options.
    """

    def __init__(self):
        super(QWidget,self).__init__()

        # parameters for display
        self.num_questions = 3
        self.name = "Robot Questions"
        # self.yes_btn_style

        self.initUI()

        rospy.Subscriber("robot_questions", Question, self.question_update)
        self.pub = rospy.Publisher("answers",Answer,queue_size=10)

    def initUI(self):
        self.container = QGroupBox('Robot Questions')
        size_policy = self.container.sizePolicy()
        size_policy.setVerticalPolicy(QSizePolicy.Expanding)
        size_policy.setHorizontalPolicy(QSizePolicy.Expanding)
        self.container.setSizePolicy(size_policy)
        self.container.setStyleSheet(groupbox_style)
        # self.container.setStyleSheet(PullQuestion_style)
        self.container_layout = QVBoxLayout()
        self.container_layout.addWidget(self.container)


        self.previous_q_container = QGroupBox()
        self.prev_q_layout = QVBoxLayout()
        self.container_layout.addWidget(self.previous_q_container)
        self.previous_q_container.setLayout(self.prev_q_layout)

        self.last_question = QLabel("Last question was: ")
        self.last_question.setStyleSheet(question_text_style)
        self.prev_q_layout.addWidget(self.last_question)

        self.last_answer = QLabel("Last answer was: ")
        self.last_answer.setStyleSheet(question_text_style)
        self.prev_q_layout.addWidget(self.last_answer)

        # main widget layout
        self.main_layout = QVBoxLayout()

        # self.name_label = QLabel(self.name)
        # self.main_layout.addWidget(self.name_label)

        self.make_question_fields()

        self.container.setLayout(self.main_layout)
        self.setLayout(self.container_layout)

    def make_question_fields(self):
        """
        Creates the question widgets to be added to the question list.
        The question widget contains the text of the question, a yes button, and
        a no button. These buttons are connected to yes no question slots.
        """
        self.question_fields = []

        for i in range(0,self.num_questions):
            question_field = PullQuestion()
            self.question_fields.append(question_field)
            self.main_layout.addWidget(question_field)
            # question_field.hide()

    def get_questions(self,qids):
        """
        For transmitted integer question IDs, get the corresponding questions
        from the question list, and return a dictionary.
        """

        question_list = []
        for qid in qids:
            question_list.append(questions[qid])

        return question_list

    def scale_VOI_magnitude(self,weights):
        """
        Scale the magintudes of the VOI wrt the maximum weight
        """
        max_value = 100
        scale_factor = max_value / max(weights)
        weights = [weight*scale_factor for weight in weights]
        return weights

    def get_new_question(self):
        """
        Get new next most valuable question to display if a question is answered
        'I don't know'.
        """
        if self.count < len(self.questions):
            new_question = self.questions[self.count]
            new_question_weight = self.question_weights[self.count]
            new_qid = self.qids[self.count]
            self.count += 1
            return (new_qid,new_question,new_question_weight)
        else:
            return None

    def question_update(self,msg):
        """
        Update the displayed questions by getting the new questions, created the
        associated question objects, and updating the display.
        """
        print('msg received')
        self.count = 0
        self.qids = msg.qids
        self.question_weights = msg.weights
        self.questions = msg.questions
        # weights = self.scale_VOI_magnitude(weights)

        for i in range(0,self.num_questions):
            self.question_fields[i].set_question(self.qids[i],self.questions[i],
                            self.question_weights[i])
            self.count += 1
            # self.question_fields[i].show()

class PullQuestion(QWidget):

    def __init__(self,qid=0,question_text='hello'):

        super(QWidget,self).__init__()

        self.pub = rospy.Publisher("answered",Answer,queue_size=10)

        self.layout = QHBoxLayout()

        self.qid = qid

        # make sure widget doesn't resize when hidden
        size_policy = self.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        self.setSizePolicy(size_policy)

        self.text = QLabel(question_text)
        self.text.setWordWrap(True)
        self.text.setStyleSheet(question_text_style)

        self.yes_btn = QPushButton('YES')
        self.yes_btn.setSizePolicy(QSizePolicy())
        self.yes_btn.setStyleSheet(yes_btn_style)
        self.yes_btn.clicked.connect(self.answered)
        # self.yes_btn.clicked.connect(self.answer_color)

        self.no_btn = QPushButton('NO')
        self.no_btn.setSizePolicy(QSizePolicy())
        self.no_btn.setStyleSheet(no_btn_style)
        self.no_btn.clicked.connect(self.answered)
        # self.no_btn.clicked.connect(self.answer_color)

        self.null_btn = QPushButton('?')
        self.null_btn.setSizePolicy(QSizePolicy())
        self.null_btn.setStyleSheet(null_btn_style)
        self.null_btn.clicked.connect(self.answered)
        # self.null_btn.clicked.connect(self.answer_color)

        self.buttons = [self.yes_btn,self.no_btn,self.null_btn]

        # Make answer indicator
        # self.answer_rect_container = QLabel()
        # self.answer_rect_container.setStyleSheet(answer_indicator_style)
        # self.answer_rect = QPixmap(QSize(10,20))
        # self.answer_rect.fill(QColor('gray'))
        # self.answer_rect_container.setPixmap(self.answer_rect)

        # Make bar to indicate VOI of question
        # self.voi_weight = QProgressBar()
        # self.voi_weight.setSizePolicy(QSizePolicy())
        # self.voi_weight.setTextVisible(False)

        self.layout.addWidget(self.text)
        self.layout.addWidget(self.yes_btn)
        self.layout.addWidget(self.no_btn)
        self.layout.addWidget(self.null_btn)
        # self.layout.addWidget(self.answer_rect_container)
        # self.layout.addWidget(self.voi_weight)

        # self.setStyleSheet(PullQuestion_style)

        # self.layout.setContentsMargin(0,0,0,0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

        self.hide()

    def set_question(self,qid,question_text,weight):
        self.qid = qid
        self.weight = weight
        # self.voi_weight.setValue(int(weight))
        self.text.setText(question_text)
        self.show()

    def answer_color(self):
        """
        Flash color when question is answered
        """
        color = ''
        if self.sender() is self.yes_btn:
            # self.setStyleSheet("background-color: green;")
            color = 'green'
            self.answer_rect.fill(QColor(color))
            self.answer_rect_container.setPixmap(self.answer_rect)

        elif self.sender() is self.no_btn:
            # self.setStyleSheet("background-color: darkred;")
            color = 'darkred'
            self.answer_rect.fill(QColor(color))
            self.answer_rect_container.setPixmap(self.answer_rect)

        elif self.sender() is self.null_btn:
            # self.setStyleSheet("background-color: skyblue;")
            color = 'skyblue'
            self.answer_rect.fill(QColor(color))
            self.answer_rect_container.setPixmap(self.answer_rect)

        # w = QWidget()
        # self.show()
        # time.sleep(1)
        # self.setStyleSheet("background-color: lightgray")

    def answered(self):
        """
        When a question is answered, publish that answer to ROS topic and hide
        the question field until next update
        """
        # hide the question
        self.hide()
        # determine answer based on which button was clicked
        ans = None
        ans_text = None
        if self.sender() is self.yes_btn:
            ans = 1
            ans_text = 'Yes'
        elif self.sender() is self.no_btn:
            ans = 0
            ans_text = 'No'
        elif self.sender() is self.null_btn:
            ans_text = 'I don\'t know'
        self.parentWidget().parentWidget().last_question.setText('Last question was: ' + self.sender().parentWidget().text.text())
        self.parentWidget().parentWidget().last_answer.setText('Last answer was: ' + ans_text)
        question = self.parentWidget().parentWidget().get_new_question()
        if question is not None:
            self.set_question(question[0],question[1],question[2])

        # create answer ROS message
        if ans is not None:
            msg = Answer()
            msg.qid = self.qid
            msg.ans = ans
            # publish answer
            self.pub.publish(msg)




robots = ["Deckard","Roy","Pris","Zhora"]

targets = ["nothing","a robber","Roy","Pris","Zhora"]

certainties = ["I know"]

positivities = ["is", "is not"]

object_relations = ["behind","in front of","left of","right of","near"]

objects = ["the bookcase","the cassini poster","the chair","the checkers table",
            "the desk","the dining table","the fern","the filing cabinet",
            "the fridge","the mars poster","Deckard"]

area_relations = ["inside","near","outside"]

areas = ["the study","the billiard room","the hallway","the dining room",
            "the kitchen","the library"]

movement_types = ["moving","stopped"]

movement_qualities = ["slowly","moderately","quickly"]

class HumanPush(QWidget):
    """
    The human push questions widget. At any time the user can construct a
    sentence from a codebook of words and push that observation to the robot.
    """

    def __init__(self):
        super(QWidget,self).__init__()

        self.name = "Human Observations"

        self.pub = rospy.Publisher("human_push",String,queue_size=10)

        self.initUI()

    def initUI(self):
        self.main_and_title = QVBoxLayout()
        self.main_layout = QHBoxLayout()

        # add name as a label
        self.name_label = QLabel(self.name)
        self.name_label.setStyleSheet(widget_title_style)
        self.main_and_title.addWidget(self.name_label)

        self.main_and_title.addLayout(self.main_layout)

        # make tab container
        self.tabs = QTabWidget()
        self.position_objects_tab = QWidget()
        self.position_objects_tab.layout = QHBoxLayout()
        self.position_area_tab = QWidget()
        self.position_area_tab.layout = QHBoxLayout()
        self.movement_tab = QWidget()
        self.movement_tab.layout = QHBoxLayout()
        self.tabs.addTab(self.position_objects_tab,'Position (Objects)')
        self.tabs.addTab(self.position_area_tab,'Position (Area)')
        self.tabs.addTab(self.movement_tab,'Movement')

        # add tabs to main layout
        self.main_layout.addWidget(self.tabs)

        # self.codebook1 = QListWidget()
        # self.codebook2 = QListWidget()
        # self.codebook3 = QListWidget()
        # self.codebook4 = QListWidget()
        # self.codebook5 = QListWidget()
        # self.main_layout.addWidget(self.codebook1)
        # self.main_layout.addWidget(self.codebook2)
        # self.main_layout.addWidget(self.codebook3)
        # self.main_layout.addWidget(self.codebook4)
        # self.main_layout.addWidget(self.codebook5)

        self.widget_list = []
        # make Position Objects codebook
        object_boxes = [certainties,targets,positivities,object_relations,
                            objects]
        object_layout, object_widget_list = self.make_codebook(object_boxes,
                                                self.position_objects_tab.layout)
        self.position_objects_tab.setLayout(object_layout)
        self.widget_list.append(object_widget_list)

        # make Position Area codebook
        area_boxes = [certainties,targets,positivities,area_relations,areas]
        area_layout, area_widget_list = self.make_codebook(area_boxes,
                                            self.position_area_tab.layout)
        self.position_area_tab.setLayout(area_layout)
        self.widget_list.append(area_widget_list)

        # make Movement codebook
        movement_boxes = [certainties,targets,positivities,movement_types,
                            movement_qualities]
        movement_layout, movement_widget_list = self.make_codebook(movement_boxes,
                                                    self.movement_tab.layout)
        self.movement_tab.setLayout(movement_layout)
        self.widget_list.append(movement_widget_list)

        # add the question parts to the codebooks
        # self.add_list_items()

        # make the 'send' and 'clear' buttons
        self.send_btn = QPushButton('Send')
        self.send_btn.clicked.connect(self.publish_msg)
        self.send_btn.setStyleSheet(send_btn_style)
        self.clear_btn = QPushButton('Clear')
        self.clear_btn.clicked.connect(self.clear_selection)
        self.clear_btn.setStyleSheet(clear_btn_style)

        # make layout for 'send' and 'clear' buttons
        self.btn_column = QVBoxLayout()
        self.btn_column.addWidget(self.clear_btn)
        self.btn_column.addWidget(self.send_btn)
        self.main_layout.addLayout(self.btn_column)

        # self.setSizePolicy(QSizePolicy())

        self.setLayout(self.main_and_title)

    def make_codebook(self,boxes,tab_widget_layout):
        """
        Make a codebook for a category of observations given the chosen items
        """
        widget_list = []
        for box in boxes:
            codebook_box = QListWidget()
            count = 1
            for item in box:
                list_item = QListWidgetItem(item)
                codebook_box.insertItem(count,list_item)
                count += 1
            tab_widget_layout.addWidget(codebook_box)
            widget_list.append(codebook_box)

        return tab_widget_layout, widget_list

    def get_answer(self):
        """
        Get the answer the user has created with the selections in the three
        codebook boxes.
        """
        # print(self.codebook1.selectedItems().isEmpty())

        # get index of selected tab
        idx = self.tabs.currentIndex()
        answer = ''

        # get selected text from all boxes in selected tab
        for codebook in self.widget_list[idx]:
            ans = ''
            try:
                ans = codebook.selectedItems()[0].text()
            except IndexError:
                error_dialog = QErrorMessage(self)
                error_dialog.showMessage('You must make a selection in all boxes before \
                                            attempting to send a message.')
                return None
            answer = answer + " " + ans
        answer.lstrip(' ')
        print(answer)
        return answer

    def clear_selection(self):
        """
        Clear the selected components of an answer when the 'CLEAR' button is
        is pressed, or when an answer is sent.
        """
        for widget in self.widget_list:
            for codebook in widget:
                codebook.clearSelection()

    def publish_msg(self):
        """
        Get the answer, clear selections and publish answer to question to
        topic over ROS.
        """
        answer = self.get_answer()
        if answer is not None:
            msg = String(answer)
            self.pub.publish(msg)

        self.clear_selection()



MapDisplay_style = "\
                        MapDisplay {   \
                            border-style: solid;   \
                            border-width: 10 px;  \
                            border-color: black; \
                        }"

class MapDisplay(QWidget):
    """
    The widget to display the cop's belief overlayed onto a map of the environment.
    """

    def __init__(self):
        super(QWidget,self).__init__()

        self.name = "Belief Map"

        rospy.Subscriber("/interface_map", Image, self.ros_update)
        self.format = QImage.Format_RGB888

        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        self.name_label = QLabel(self.name)
        self.name_label.setStyleSheet(widget_title_style)
        self.main_layout.addWidget(self.name_label)

        self.image_view = QPixmap()
        # self.image_view.load('/home/ian/catkin_ws/src/cops-and-robots-2.0/observation_interface/src/Clue_map.png')
        # self.image_view = self.image_view.scaled(600,450,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.pic_label = QLabel(self)
        # self.pic_label.setScaledContents(True)
        # self.pic_label.setPixmap(self.image_view)
        self.main_layout.addWidget(self.pic_label)

        # self.pic_label.setFixedSize(500,350)
        # self.setStyleSheet(MapDisplay_style)
        self.setLayout(self.main_layout)
        self.show()

    def ros_update(self, msg):
        image_data = msg.data
        image_height = msg.height
        image_width = msg.width
        bytes_per_line = msg.step
        self.image = QImage(image_data,image_width,image_height,bytes_per_line,self.format)

        if not self.image.isNull():
            self.pic_label.setPixmap(QPixmap.fromImage(self.image))
        else:
            print("{} sent bad frames!".format(self.name))

class VideoContainer(QWidget):
    """
    A general class for widgets to display video feeds.
    """

    def __init__(self):
        super(QWidget,self).__init__()
        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        self.name_label = QLabel(self.name)
        self.name_label.setStyleSheet(widget_title_style)
        self.main_layout.addWidget(self.name_label)

        self.counter = 0

        self.canvas = VideoCanvas(self.size)
        self.canvas.setMinimumSize(QSize(*self.size))

        # self.scene = QGraphicsScene()
        # self.image_view = QGraphicsView(self.scene)

        # self.video = VideoDisplayWidget()
        # self.surface = self.video.surface

        self.image = QImage()

        # self.image_view = QPixmap(*self.size)
        # self.pic_label = QLabel(self)
        # self.pic_label.setScaledContents(True)
        # self.image_view.fill(QColor('black'))
        # self.pic_label.setPixmap(self.image_view)
        # self.canvas_widget = QWidget()
        # self.canvas = QRect()
        # self.image = QImage()
        # self.painter = QPainter(self)

        # self.main_layout.addWidget(self.video)
        self.main_layout.addWidget(self.canvas)

        # self.image.setFixedSize(QSize(self.size[0],self.size[1]))
        # self.setStyleSheet(MapDisplay_style)
        self.setLayout(self.main_layout)

    def ros_update(self,msg):
        """
        Callback function to display a ROS image topic with streaming video data.
        """
        self.counter += 1

        fmt = '>' + str((self.size[0]*self.size[1]))

        image_data = msg.data
        # l = array.array('H')
        # l.fromstring(image_data)
        # data_ordered = [struct.pack('<I',x) for x in l]
        # data_ordered = struct.pack('')
        # data_unpacked = struct.unpack(fmt,image_data)
        # print(data_unpacked)
        image_height = msg.height
        image_width = msg.width
        bytes_per_line = msg.step
        # if self.counter % 2 == 0:
        self.image = QImage(image_data,image_width,image_height,bytes_per_line,self.format)
        if not self.image.isNull():
            self.canvas.image = self.image
        self.canvas.repaint()
        # self.present_image(image)
        # if not self.image.isNull():
        #         # self.repaint()
        #         # self.pic_label.setPixmap(QPixmap.fromImage(self.image))
        #     self.canvas.image = self.image
        # else:
        #     print("{} sent bad frames!".format(self.name))
        # if not self.image.isNull():
            # self.scene.addItem(self.image)
        # else:
            # print("{} sent bad frames!".format(self.name))
        # self.image_view.show()
        # if not self.image.isNull():
        #     self.pic_label.setPixmap(QPixmap.fromImage(self.image))
        # else:
        #     print("{} sent bad frames!".format(self.name))
        # self.painter.begin(self.image_view)
        # self.painter.drawImage(self.rect(),self.image,self.rect())
        # self.painter.end()

        # self.image.show()

    # def paintEvent(self,event):
    #     # if self.image.isNull():
    #         # print('{} sent bad frames!'.format(self.name))
    #         # return
    #     painter = QPainter(self)
    #     if not self.image.isNull():
    #         painter.drawImage(self.rect(),self.image)
    #     else:
    #         print('Null image')
    #         return

    def present_image(self,image):
        frame = QVideoFrame(image)

        if not frame.isValid():
            return False

        current_format = self.surface.surfaceFormat()

        if (frame.pixelFormat() != current_format.pixelFormat()) \
            or (frame.size() != current_format.frameSize()):

            format_ = QVideoSurfaceFormat(frame.size(),frame.pixelFormat())

            if not self.surface.start(format_):
                print("surface not started")
                return False

        if not self.surface.present(frame):
            print("no present")
            self.surface.stop()
            return False
        else:
            self.surface.present(frame)
            self.show()
            return True

class VideoCanvas(QWidget):
    """
    Widget to paint video feed image frames to
    """
    def __init__(self,size):
        super(VideoCanvas,self).__init__()
        self.size = size
        self.image = None

    def paintEvent(self,event):
        painter = QPainter(self)
        if (self.image is not None) and (not self.image.isNull()):
            painter.drawImage(self.rect(),self.image)



class CopVideo(VideoContainer):
    """
    Subclasses VideoDisplay to display the cop's camera feed.
    """

    def __init__(self,cop_name='pris'):
        super(VideoContainer,self).__init__()
        self.name = "Cop Video"
        self.topic_name = '/' + cop_name + '/camera/rgb/image_color'
        self.size = (500,350)
        self.img = 'placeholder.png'
        self.format = QImage.Format_RGB888
        self.initUI()
        rospy.Subscriber(self.topic_name, Image,self.ros_update)

class SecurityCamera(VideoContainer):
    """
    Subclasses VideoDisplay to display the feed of a security camera.

    """

    def __init__(self,num,location):
        super(VideoContainer,self).__init__()
        self.name = "Camera {}: {}".format(num,location)
        self.topic_name = 'cam{}'.format(num)
        self.topic_name = "/" + self.topic_name + "/usb_cam/image_raw"
        self.size = (320,240)
        self.img = 'smaller_placeholder.png'
        self.initUI()
        self.format = QImage.Format_RGB888
        rospy.Subscriber(self.topic_name, Image, self.ros_update)
