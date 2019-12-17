#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:16:21 2019

@author: eva
"""

import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, list_IDs, sampling, subbatch): #([fat, wat, dX, dY, label, label_sizes])
        '_subbatch_: how many patches, sampled according to _sampling_ strategy will be extracted per subject/image'
        #self.labels = list_IDs[:, 4]
        #self.masks = list_IDs[:, 5]
        #self.slike = list_IDs[:, 0:2]
        #self.dists = list_IDs[:, 2:4]
        self.list_IDs = list_IDs[:, 0:-1]
        self.label_sizes = list_IDs[:, -1]
        self.sampling = sampling
        self.subbatch = max(subbatch, sum(sampling)) #subbatch should be divisible by nr of samples wanted per class

    def __len__(self):
        'total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, item):
        'get one sample: a subbatch of patches from 1 subject'
        #select subject
        subj = self.list_IDs[item]

        #get center points according to sampling
        