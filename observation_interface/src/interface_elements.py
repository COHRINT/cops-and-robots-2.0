#!/usr/bin/env python

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

import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QFont, QPixmap

questions = ["What is your favorite color?",
            "What is your quest?",
            "What is the airspeed of an unladen swallow?"]

PullQuestion_style = "\
                        .QWidget {   \
                            border-style: solid;   \
                            border-width: 1px;  \
                        }"

yes_btn_style = "\
                QPushButton {   \
                    color: white;   \
                    background-color: green;    \
                }"

no_btn_style = "\
                QPushButton {   \
                    color: white;   \
                    background-color: darkred;  \
                }"

null_btn_style = "\
                    QPushButton {   \
                        color: black;   \
                        background-color: skyblue;  \
                    }"

question_text_style = "\
                        QLabel {    \
                            font: bold; \
                        }"


class RobotPull(QWidget):
    """
    The robot pull questions widget. Displays questions chosen by robot and their
    associated value of information, and presents user with 'yes' and 'no' options.
    """

    def __init__(self):
        super(QWidget,self).__init__()

        # parameters for display
        self.num_questions = 5
        self.name = "Robot Questions"
        # self.yes_btn_style

        self.initUI()

    def initUI(self):
        # self.make_buttons()

        # main widget layout
        self.main_layout = QVBoxLayout()

        self.name_label = QLabel(self.name)
        self.main_layout.addWidget(self.name_label)

        self.make_question_fields()

        self.setLayout(self.main_layout)


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


    def get_questions(self,qids):
        """
        For transmitted integer question IDs, get the corresponding questions
        from the question list, and return a dictionary.
        """

        question_list = []
        for qid in qids:
            question_list.append(questions[qid])

        return question_list

    def update(self,qids):
        """
        Update the displayed questions by getting the new questions, created the
        associated question objects, and updating the display.
        """
        for i in range(0,len(qids)):
            self.question_fields[i].text.setText(questions[qids[i]])
        # question_list = self.get_questions(qids)

    def question_answered(self):
        pass

class PullQuestion(QWidget):

    def __init__(self,question_text='hello'):

        super(QWidget,self).__init__()

        self.layout = QHBoxLayout()

        self.text = QLabel(question_text)
        self.text.setWordWrap(True)
        self.text.setStyleSheet(question_text_style)

        self.yes_btn = QPushButton('YES')
        self.yes_btn.setStyleSheet(yes_btn_style)
        self.yes_btn.clicked.connect(self.answered)

        self.no_btn = QPushButton('NO')
        self.no_btn.setStyleSheet(no_btn_style)
        self.no_btn.clicked.connect(self.answered)

        self.null_btn = QPushButton('I DON\'T KNOW')
        self.null_btn.setStyleSheet(null_btn_style)
        self.null_btn.clicked.connect(self.answered)

        self.buttons = [self.yes_btn,self.no_btn,self.null_btn]

        self.layout.addWidget(self.text)
        self.layout.addWidget(self.yes_btn)
        self.layout.addWidget(self.no_btn)
        self.layout.addWidget(self.null_btn)

        self.setLayout(self.layout)

    def set_text(self,question_text):
        self.text.setText(question_text)

    def answered(self):
        """
        When a question is answered, publish that answer to ROS topic and hide
        the question field until next update
        """
        self.hide()

        # self.publish()




first_box = ["The red robot",
            "The green one",
            "Deckard",
            "Me"]

second_box = ["is moving towards",
            "is near",
            "is by",
            "is next to",
            "is inside"]

third_box = ["the kitchen.",
            "the fern.",
            "the table.",
            "the hallway."]




class HumanPush(QWidget):
    """
    The human push questions widget. At any time the user can construct a
    sentence from a codebook of words and push that observation to the robot.
    """

    def __init__(self):
        super(QWidget,self).__init__()

        self.name = "Human Questions"

        self.initUI()

    def initUI(self):
        self.main_and_title = QVBoxLayout()
        self.main_layout = QHBoxLayout()
        self.main_and_title.addLayout(self.main_layout)

        # add name as a label
        self.name_label = QLabel(self.name)
        self.main_layout.addWidget(self.name_label)

        # make codebook compoments
        self.codebook1 = QListWidget()
        self.codebook2 = QListWidget()
        self.codebook3 = QListWidget()
        self.codebook4 = QListWidget()
        self.codebook5 = QListWidget()
        self.main_layout.addWidget(self.codebook1)
        self.main_layout.addWidget(self.codebook2)
        self.main_layout.addWidget(self.codebook3)
        self.main_layout.addWidget(self.codebook4)
        self.main_layout.addWidget(self.codebook5)

        # add the question parts to the codebooks
        self.add_list_items()

        # make the 'send' and 'clear' buttons
        self.send_btn = QPushButton('Send')
        self.send_btn.clicked.connect(self.publish_msg)
        self.clear_btn = QPushButton('Clear')
        self.clear_btn.clicked.connect(self.clear_selection)

        # make layout for 'send' and 'clear' buttons
        self.btn_column = QVBoxLayout()
        self.btn_column.addWidget(self.clear_btn)
        self.btn_column.addWidget(self.send_btn)
        self.main_layout.addLayout(self.btn_column)

        self.setLayout(self.main_and_title)

    def add_list_items(self):
        """
        Add parts of questions to the appropriate codebook widget to be displayed.
        """
        count = 1
        for item in first_box:
            list_item = QListWidgetItem(item)
            self.codebook1.insertItem(count,list_item)
            count += 1

        count = 1
        for item in second_box:
            list_item = QListWidgetItem(item)
            self.codebook2.insertItem(count,list_item)
            count += 1

        count = 1
        for item in third_box:
            list_item = QListWidgetItem(item)
            self.codebook3.insertItem(count,list_item)
            count += 1

        count = 1
        for item in third_box:
            list_item = QListWidgetItem(item)
            self.codebook4.insertItem(count,list_item)
            count += 1

        count = 1
        for item in third_box:
            list_item = QListWidgetItem(item)
            self.codebook5.insertItem(count,list_item)
            count += 1

    def get_answer(self):
        """
        Get the answer the user has created with the selections in the three
        codebook boxes.
        """
        pass

    def clear_selection(self):
        """
        Clear the selected components of an answer when the 'CLEAR' button is
        is pressed, or when an answer is sent.
        """
        self.codebook1.clearSelection()
        self.codebook2.clearSelection()
        self.codebook3.clearSelection()
        self.codebook4.clearSelection()
        self.codebook5.clearSelection()

    def publish_msg(self):
        """
        Get the answer, clear selections and publish answer to question to
        topic over ROS.
        """
        answer = self.get_answer()
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

        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        self.name_label = QLabel(self.name)
        self.main_layout.addWidget(self.name_label)

        self.image_view = QPixmap('placeholder.png')
        self.pic_label = QLabel(self)
        self.pic_label.setScaledContents(True)
        self.pic_label.setPixmap(self.image_view)
        self.main_layout.addWidget(self.pic_label)

        self.pic_label.setFixedSize(700,350)
        # self.setStyleSheet(MapDisplay_style)
        self.setLayout(self.main_layout)
        # self.show()

    def update(self):
        pass

class VideoDisplay(QWidget):
    """
    A general class for widgets to display video feeds.
    """

    def __init__(self):
        super(QWidget,self).__init__()
        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        self.name_label = QLabel(self.name)
        self.main_layout.addWidget(self.name_label)

        self.image_view = QPixmap(self.img)
        self.pic_label = QLabel(self)
        self.pic_label.setScaledContents(True)
        self.pic_label.setPixmap(self.image_view)
        self.main_layout.addWidget(self.pic_label)

        self.pic_label.setFixedSize(QSize(self.size[0],self.size[1]))
        # self.setStyleSheet(MapDisplay_style)
        self.setLayout(self.main_layout)

class CopVideo(VideoDisplay):
    """
    Subclasses VideoDisplay to display the cop's camera feed.
    """

    def __init__(self):
        super(VideoDisplay,self).__init__()
        self.name = "Cop Video"
        self.size = (700,350)
        self.img = 'placeholder.png'
        self.initUI()

class SecurityCamera(VideoDisplay):
    """
    Subclasses VideoDisplay to display the feed of a security camera.
    """

    def __init__(self,num,location):
        super(VideoDisplay,self).__init__()
        self.name = "Camera {}: {}".format(num,location)
        self.size = (300,300)
        self.img = 'smaller_placeholder.png'
        self.initUI()
