#!/usr/bin/env python

import numpy as np
from map_maker import Map
import re

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

# x = np.load('likelihoods.npy')

statements = []


def gen_statements():
    for atom in certainties:
        target = "Roy"
        for qual in positivities:
            for relation in object_relations:
                for object_ in objects:
                    statement = atom + " " + target + " " + qual + " " + relation \
                                    + " " + object_
                    statements.append(statement)
            for relation in area_relations:
                for area in areas:
                    statement = atom + " " + target + " " + qual + " " + relation \
                                    + " " + area
                    statements.append(statement)
    print(statements)
    return statements

def gen_mapdict(statements):
    map_dict = {}

    for i, statement in enumerate(statements):
        for i, question in enumerate(x['question']):
            print('{}\t - {}'.format(i,question))
        print(statement)
        qid = raw_input('Enter matching question id or -1 to skip:  ')
        qid = int(qid)
        if qid is -1:
            continue
        pos = raw_input('Enter 1 for pos, 0 for neg:  ')
        pos = bool(pos)
        map_dict[statement] = [x['question'][qid],pos]

    print(map_dict)
    return map_dict

def gen_questions(yaml_file):
    room_rel = ["inside"]
    obj_rel = ["in front of","behind","left of","right of"]


    m = Map(yaml_file)
    # questions = {'rooms':[],'objects':{}}
    questions = []
    i = 0
    for room in m.rooms:
        print room
        questions.append([])
        for rel in room_rel:
            question = 'Is Roy ' + rel + ' the ' + room + '?'
            questions[i].append(question)
        for obj in m.rooms[room]['objects']:
            for rel in obj_rel:
                question = 'Is Roy ' + rel + ' the ' + obj + '?'
                questions[i].append(question)
        i+=1

    # for obj in m.objects:
    #     if not re.search('wall',obj):
    #         questions['objects'][obj] = []
    #         for rel in obj_rel:
    #             question = 'Is Roy ' + rel + ' ' + 'the ' + obj + '?'
    #             questions['objects'][obj].append(question)

    # print questions['rooms']
    print '\n *** \n'
    print questions

    return questions

if __name__ == "__main__":
    gen_questions('map2.yaml')
