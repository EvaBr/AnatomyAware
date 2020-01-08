#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:17:05 2019

@author: eva
"""

import glob
from generators import *
#from helpers import *
from networks import *

subjekti = glob.glob('/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAware/TRAINdata/sub*'); subjekti.sort()
labele = glob.glob('/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAware/TRAINdata/lab*'); labele.sort()
dataset = POEMDataset(subjekti, labele, 25, [1,1,1,1,1,1])
train_loader = data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=my_collate)



#for i, batch in enumerate(train_loader):
#    print(i, batch[0].shape) #or rather, plot sth?