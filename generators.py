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
class POEMDatasetTEST(data.Dataset): #TODO
    def __init__(self, list_IDs,  channels, subsampled=False, channels_sub=None, input2=None, channels2=None): #([fat, wat, dX, dY, label, label_sizes])
        """channels - list of which (of the 4) channels to give as normal (and possibily subsampled)
        input. channels2 - which channels to give to second input, if input2 not None.
        input2 - patchsize of second input. """
        self.list_IDs = list_IDs
        self.channels = channels
        self.subsample = subsampled
        self.subchannels = channels_sub
        self.input2 = input2 #should also be odd number!!!
        self.channels2 = channels2


    def __len__(self):
        'total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, item):
        'get one patch'
        patch_name = self.list_IDs[item]
        patch = np.load(patch_name)

        out = [torch.from_numpy(patch[self.channels, ...])]

        if self.subsample:
            patch_subs = np.load("_subsmpl.".join(patch_name.split(".")))
            out.append(torch.from_numpy(patch_subs[self.subchannels, ...]))
        if self.input2:
            s = (patch.shape[-1] - self.input2)//2 #assume patches as saved in memory are cubes!!!
            out.append(torch.from_numpy(patch[self.channels2, s:-s, s:-s, s:-s]))

        out.append("_OUT.".join(patch_name.split(".")))
        return out


def test_collate(batch):
    slike = torch.stack([item[0].float() for item in batch], dim=0) #now you dont have subbatches, so every item in your batch is already individ. image
    names = np.array([item[-1] for item in batch])
    out = [slike]

    for i in range(1, len(batch[0]) - 1):
        out.append(torch.stack([item[i].float() for item in batch], dim=0))

    return [out, names]



class POEMDataset(data.Dataset): #TODO : dodaj moznost channeljev - ie da nena loadas dist.mapov
    def __init__(self, list_IDs, list_labels, segment_size, out_size, sampling):  #([fat, wat, dX, dY, label, label_sizes])
        self.list_IDs = list_IDs
        self.labels = list_labels
        self.sampling = sampling
        self.subbatch = sum(sampling) #subbatch_: how many patches, sampled according to _sampling_ strategy will be extracted per subject/image
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
        #print("centers: ", centres)
        centres = [ctr for ctr in centres if ctr]
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
    def __init__(self, list_IDs, list_labels, sampling, segment_size=25, channels=[0,1], subsample=True,
                 channels_sub=[0,1,2,3], segment_size2=None, channels2=None): #([fat, wat, dX, dY, label, label_sizes])
        """CAN BE USED as 1 pathway, or classic DM (normal+subsampled), or 2pathway, or 2pathway+subsampled.
            list_id, list_labels: lists of strings of names of files
            segment_size=25: size of training segments
            channels: list of channels to include in the normal pathway
            subsample: bool. do you want to use subsampled pathway of DM
            sampling: sampling strategy, list of size nr.channels
            channels_sub: list of channels to use for subsampled pathway
            segment_size2: size of patches for second second input, None if no second input used
            channels2: which channels to use for second input, if in use."""

        #COMMENTS: We hardcode some DeepMed details, eg outsize=segment_size-16, subsampled_seg_size = seg_size+2*16-2,
        #           subsampling factor=3, ...
        # Also hardcoded are poem specifics. Such as nrclasses = 6
        self.list_IDs = list_IDs
        self.labels = list_labels
        self.sampling = sampling
        self.subbatch = sum(sampling)
        self.segment = int((segment_size-1)/2)
        self.channels = channels
        self.labseg = int((segment_size-16-1)/2)   # x + N ->(:3)-> (x+2)/3+(N-2)/3 ->(-16)-> (x+2)/3-16+(N-2)/3 ->(*3)-> x+2+(N-2)/3*3-16*3 ==x-16
                                                                                            # (N-2)/3*3 = 2*16 - 2
        self.subsample = subsample
        self.subchannels = channels_sub
        self.subsegment = 27  #obv should be bigger than segment.
        #bigsegmentsize should be divisible by 3, as we downsample it. But also odd, to contain the center pixel.
        self.f = 3 #subsample factor. can be also input? But: bigsegment%%f=0, and bigsegm/f==odd.

        self.in2 = False if segment_size2==None else True
        self.in2seg = segment_size2
        self.in2ch = channels2

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
        bigsegment = self.segment
        if self.subsample:
            bigsegment = self.subsegment
        centres = []
        for org in range(6):
            organ = [c for c in subj[4+org] if (c[1]<s[1]-bigsegment and c[1]>=bigsegment and
                                                c[0]<s[0]-bigsegment and c[0]>=bigsegment and
                                                c[2]<s[2]-bigsegment and c[2]>=bigsegment)]
            #rows = np.random.choice(subj[10][org], size=localSampling[org], replace=False)
            #centres.append(subj[4 + org][rows])
            np.random.shuffle(organ)
            centres.append(organ[0:localSampling[org]])

        centres = [ctr for ctr in centres if ctr]
        centres = np.concatenate(centres, axis=0)
        #now sample:
        patches =  []
        patchsub = []
        patches2 = []
        truths  =  []
        #Fat,Wat,Xdist,Ydist = subj[0], subj[1], subj[2], subj[3]
        for center in centres:
           # print("center") # TODO! Why sometimes more centres than other times?? aja, verjetno zato ker je zarad bugeca moj bigsegment bil huge...
            i,j,k = center
            patches.append(torch.from_numpy(np.stack([
                subj[ch][i-self.segment:i+self.segment+1,
                        j-self.segment:j+self.segment+1,
                        k-self.segment:k+self.segment+1] for ch in self.channels], axis=0)))
            truths.append(torch.from_numpy(
                label[i - self.labseg:i + self.labseg + 1,
                    j - self.labseg:j + self.labseg + 1,
                    k - self.labseg:k + self.labseg + 1]))

        out = [torch.stack(patches)]
       # print(out[0].shape)

        if self.subsample:
            for center in centres:
                i, j, k = center
                patchsub.append(torch.from_numpy(np.stack([
                    subj[ch][i-self.subsegment:i+self.subsegment+1:self.f,
                            j - self.subsegment:j + self.subsegment + 1:self.f,
                            k - self.subsegment:k + self.subsegment + 1:self.f] for ch in self.subchannels], axis=0)))
                # axis 0 bcs in pytorch channels first. so for each patch we append channels x sgm x sgm x sgm array.
            out.append(torch.stack(patchsub))
    #        print(out[-1].shape)
        if self.in2:
            for center in centres:
                i, j, k = center
                patches2.append(torch.from_numpy(np.stack([
                    subj[ch][i - self.in2seg:i + self.in2seg + 1:self.f,
                            j - self.in2seg:j + self.in2seg + 1:self.f,
                            k - self.in2seg:k + self.in2seg + 1:self.f] for ch in self.in2ch], axis=0)))
            out.append(torch.stack(patches2))

        out.append(torch.stack(truths))
        return out


def multi_input_collate(batch):
    data = torch.cat([item[0] for item in batch], dim=0)
    target = torch.cat([item[-1] for item in batch], dim=0)
    out = [data]

    for i in range(1, len(batch[0])-1):
        out.append(torch.cat([item[i] for item in batch], dim=0))

    return [out, target] #is this structure ok for networks??