#!/usr/bin/env python

import numpy as np

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

x = np.load('likelihoods.npy')

statements = []

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
