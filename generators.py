#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:16:21 2019

@author: eva
"""

import torch
from torch.utils import data
import pickle
import numpy as np
import re
#from torchvision.transforms import ToTensor
class POEMDatasetTEST(data.Dataset): #TODO FIX
    def __init__(self, list_IDs,  patchsize=50, overlap=16): #([fat, wat, dX, dY, label, label_sizes])
        '_subbatch_: how many patches, sampled according to _sampling_ strategy will be extracted per subject/image'
        self.list_IDs = list_IDs
        self.patchsize = patchsize #it's great if patchsize divides 256-overlap. Otherwise last batch might be smaller...
        self.countermax = (256-overlap)//patchsize
        self.overlap = overlap #how much patches need to overlap, per dim, as they get reduced by hte network. good if divides 256.
        #eg for deep medic, we get 25 -> 9, so 16 pixels on each dim get removed.

    def __len__(self):
        'total number of samples'
        return len(self.list_IDs)*self.countermax**2

    def __getitem__(self, item):
        'get one sample: larger part of 1 subject'
        counter = item % (self.countermax**2) #counts which nr of patch we are extracting
        counterx = counter//self.countermax
        counterz = counter%self.countermax

        ajtem = item//(self.countermax**2)
        subj = pickle.load(open(self.list_IDs[ajtem], 'rb'))
        name = np.array(re.findall(r'.*subj([0-9]*)\.pickle', self.list_IDs[ajtem]), dtype=np.int16)

        #now cut it to appropriate nr. of pieces and take the piece nr. self.counter.
        s = subj[0].shape #(256, x, 256)
        subj = np.stack([subj[0], subj[1], subj[2], subj[3]])
        #print(subj.dtype, label.dtype, name.dtype)

        #we will cut by first and last dimension:
        subj = subj[:, (self.patchsize-self.overlap)*counterx:(self.patchsize-self.overlap)*(counterx+1)+self.overlap,
                    :, (self.patchsize-self.overlap)*counterz:(self.patchsize-self.overlap)*(counterz+1)+self.overlap]

        where_it_belongs = np.array([counterx, counterz])

        return torch.from_numpy(subj), torch.from_numpy(name), torch.from_numpy(where_it_belongs) #torch.from_numpy(label[1:-1, 1:-1, 1:-1]), torch.from_numpy(name)


def test_collate(batch): #TODO FIX
    slike = torch.cat([item[0] for item in batch], dim=0)
    names = torch.cat([item[1] for item in batch], dim=0)
    placing = torch.cat([item[2] for item in batch], dim=0)
    return [slike, names, placing]



class POEMDataset(data.Dataset): #TODO : dodaj moznost channeljev - ie da nena loadas dist.mapov
    def __init__(self, list_IDs, list_labels, segment_size, out_size, sampling): #([fat, wat, dX, dY, label, label_sizes])
        '_subbatch_: how many patches, sampled according to _sampling_ strategy will be extracted per subject/image'
        self.list_IDs = list_IDs
        self.labels = list_labels
        self.sampling = sampling
        self.subbatch = sum(sampling)
        self.segment = int((segment_size-1)/2)
        self.labseg = int((out_size-1)/2)

    def __len__(self):
        'total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, item):
        'get one sample: a subbatch of patches from 1 subject'
        #select subject
        subj = pickle.load(open(self.list_IDs[item], 'rb'))
        label = pickle.load(open(self.labels[item], 'rb'))
        #get patches according to sampling strategy.
        #first let's correct sampling for the given subj:
        localSampling = self.sampling
        for org in range(6): #subj[10]:
            razlika = self.sampling[org]-subj[10][org]
            if 0<razlika:
                localSampling[org] = subj[10][org]
                localSampling[0] += razlika
        #now get center points, careful with where you are allowed to sample:
        s = label.shape
        #print(subj[0].shape)
        #print(s)
        centres = []
        for org in range(6):
            organ = [c for c in subj[4+org] if (c[1]<s[1]-self.segment and c[1]>=self.segment and
                                                c[0]<s[0]-self.segment and c[0]>=self.segment and
                                                c[2]<s[2]-self.segment and c[2]>=self.segment)]
            #rows = np.random.choice(subj[10][org], size=localSampling[org], replace=False)
            #centres.append(subj[4 + org][rows])
            np.random.shuffle(organ)
            centres.append(organ[0:localSampling[org]])

        centres = np.concatenate(centres, axis=0)
        #now sample:
        patches= []
        truths = []
        #Fat,Wat,Xdist,Ydist = subj[0], subj[1], subj[2], subj[3]
        for center in centres:
            i,j,k = center
            patches.append(torch.from_numpy(np.stack([
                subj[0][i-self.segment:i+self.segment+1, j-self.segment:j+self.segment+1, k-self.segment:k+self.segment+1],
                subj[1][i-self.segment:i+self.segment+1, j-self.segment:j+self.segment+1, k-self.segment:k+self.segment+1],
                subj[2][i-self.segment:i+self.segment+1, j-self.segment:j+self.segment+1, k-self.segment:k+self.segment+1],
                subj[3][i-self.segment:i+self.segment+1, j-self.segment:j+self.segment+1, k-self.segment:k+self.segment+1],
            ], axis=0))) #axis 0 bcs in pytorch channels first. so for each patch we append 4 x sgm x sgm x sgm array.
            truths.append(torch.from_numpy(
                label[i-self.labseg:i+self.labseg+1, j-self.labseg:j+self.labseg+1, k-self.labseg:k+self.labseg+1]))
        #for p in patches: print(p.shape)

        return torch.stack(patches), torch.stack(truths)


def my_collate(batch):
    data = torch.cat([item[0] for item in batch], dim=0)
    target = torch.cat([item[1] for item in batch], dim=0)

    return [data, target]

#Ok, dataloader extremely slow, but at least seems to work. :)
###################################################################
#import glob
#subjekti = glob.glob('/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAware/TRAINdata/sub*'); subjekti.sort()
#labele = glob.glob('/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAware/TRAINdata/lab*'); labele.sort()
#dataset = POEMDataset(subjekti, labele, 25, [1,1,1,1,1,1])
#train_loader = data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=my_collate)
#
#for i, batch in enumerate(train_loader):
#    print(i, batch[0].shape) #or rather, plot sth?
###################################################################




#now dataset for two inputs
class POEMDatasetMultiInput(data.Dataset):
    def __init__(self, list_IDs, list_labels, segment_size, big_segment_size, sampling): #([fat, wat, dX, dY, label, label_sizes])
        '_subbatch_: how many patches, sampled according to _sampling_ strategy will be extracted per subject/image'
        self.list_IDs = list_IDs
        self.labels = list_labels
        self.sampling = sampling
        self.subbatch = sum(sampling)
        self.segment = int((segment_size-1)/2)
        self.bigsegment = int(big_segment_size-1)/2 #obv should be bigger than segment.
        #bigsegmentsize should be divisible by 3, as we downsample it. But also odd, to contain the center pixel.
        self.f = 3 #subsample factor. can be also input? But: bigsegment%%f=0, and bigsegm/f==odd.

    def __len__(self):
        'total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, item):
        'get one sample: a subbatch of patches from 1 subject'
        #select subject
        subj = pickle.load(open(self.list_IDs[item], 'rb'))
        label = pickle.load(open(self.labels[item], 'rb'))
        #get patches according to sampling strategy.
        #first let's correct sampling for the given subj:
        localSampling = self.sampling
        for org in range(6): #subj[10]:
            razlika = self.sampling[org]-subj[10][org]
            if 0<razlika:
                localSampling[org] = subj[10][org]
                localSampling[0] += razlika
        #now get center points, careful with where you are allowed to sample:
        s = label.shape
        #print(subj[0].shape)
        #print(s)
        centres = []
        for org in range(6):
            organ = [c for c in subj[4+org] if (c[1]<=s[1]-self.bigsegment and c[1]>=self.bigsegment and
                                                c[0]<=s[0]-self.bigsegment and c[0]>=self.bigsegment and
                                                c[2]<=s[2]-self.bigsegment and c[2]>=self.bigsegment)]
            #rows = np.random.choice(subj[10][org], size=localSampling[org], replace=False)
            #centres.append(subj[4 + org][rows])
            np.random.shuffle(organ)
            centres.append(organ[0:localSampling[org]])

        centres = np.concatenate(centres, axis=0)
        #now sample:
        patches =  []
        patchbig = []
        truths  =  []
        #Fat,Wat,Xdist,Ydist = subj[0], subj[1], subj[2], subj[3]
        for center in centres:
            i,j,k = center
            patches.append(torch.from_numpy(np.stack([
                subj[0][i-self.segment:i+self.segment+1, j-self.segment:j+self.segment+1, k-self.segment:k+self.segment+1],
                subj[1][i-self.segment:i+self.segment+1, j-self.segment:j+self.segment+1, k-self.segment:k+self.segment+1],
                subj[2][i-self.segment:i+self.segment+1, j-self.segment:j+self.segment+1, k-self.segment:k+self.segment+1],
                subj[3][i-self.segment:i+self.segment+1, j-self.segment:j+self.segment+1, k-self.segment:k+self.segment+1],
            ], axis=0))) #axis 0 bcs in pytorch channels first. so for each patch we append 4 x sgm x sgm x sgm array.
            patchbig.append(torch.from_numpy(np.stack([
                subj[0][i-self.bigsegment:i+self.bigsegment+1:self.f,
                        j-self.bigsegment:j+self.bigsegment+1:self.f,
                        k-self.bigsegment:k+self.bigsegment+1:self.f],
                subj[1][i-self.bigsegment:i+self.bigsegment+1:self.f,
                        j-self.bigsegment:j+self.bigsegment+1:self.f,
                        k-self.bigsegment:k+self.bigsegment+1:self.f],
                subj[2][i-self.bigsegment:i+self.bigsegment+1:self.f,
                        j-self.bigsegment:j+self.bigsegment+1:self.f,
                        k-self.bigsegment:k+self.bigsegment+1:self.f],
                subj[3][i-self.bigsegment:i+self.bigsegment+1:self.f,
                        j-self.bigsegment:j+self.bigsegment+1:self.f,
                        k-self.bigsegment:k+self.bigsegment+1:self.f],
            ], axis=0)))  # axis 0 bcs in pytorch channels first. so for each patch we append 4 x sgm x sgm x sgm array.
        #for p in patchbig: print(p.shape)

        return torch.stack(patches), torch.stack(patchbig), torch.stack(truths)

def multi_input_collate(batch):
    data_in1 = torch.cat([item[0] for item in batch], dim=0)
    data_in2 = torch.cat([item[1] for item in batch], dim=0)
    target = torch.cat([item[2] for item in batch], dim=0)

    return [data_in1, data_in2, target] #is this structure ok for networks??