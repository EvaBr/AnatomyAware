#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:00:52 2020

@author: eva
"""

import torch
from torch.utils import data
import numpy as np
import re
#from torchvision.transforms import ToTensor
class POEMDatasetTEST(data.Dataset): #TODO
    def __init__(self, list_IDs, channels, subsampled=False, channels_sub=None, input2=None, channels2=None): #([fat, wat, dX, dY, label, label_sizes])
        """Dataset, appropriate for apriori cut image loading. (Eg for VAL and TEST).
        list_IDs - list of names of all patches datas to include in loading.
        channels - list of which (of the 4 available) channels to give as normal (and possibily subsampled) input. 
        subsampled - whether to produce also subsampled input.
        channels_sub - list of channels to use in the subsampled input, if applicable.
        channels2 - which channels to give to second input, if input2 not None.
        input2 - patchsize of second input. """

        self.list_IDs = list_IDs
     #   self.labels = label_IDs
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
            ss = patch.shape[-1] #assume patches as saved in memory are cubes!!!
            s = (ss - self.input2)//2 # And assume that input2 segment, if used, is <=than original segment size!
            out.append(torch.from_numpy(patch[self.channels2, s:ss-s, s:ss-s, s:ss-s]))
        
      #  if label_IDs!=None: ## needed for validation <- nope, we're using the other one for validaiton now.
      #      label = torch.from_numpy(np.load(self.label_IDs[item]))
      #      out.append(label)

        out.append("_OUT.".join(patch_name.split("."))) ##needed for testing
        return out


def test_collate(batch):
    slike = torch.stack([item[0].float() for item in batch], dim=0) #now you dont have subbatches, so every item in your batch is already individ. image
    names = np.array([item[-1] for item in batch])
    out = [slike]

    for i in range(1, len(batch[0]) - 1):
        out.append(torch.stack([item[i].float() for item in batch], dim=0))

    return [out, names]




#now dataset for training with multiple inputs. No predefined cutting of patches.
class POEMDatasetMultiInput(data.Dataset):
    def __init__(self, list_IDs, list_labels, sampling, segment_size=25, channels=[0,1], subsample=True,
                 channels_sub=[0,1,2,3], segment_size2=None, channels2=None, num_classes=7): #([fat, wat, dX, dY, label, label_sizes])
        """CAN BE USED for 1 pathway, or classic DM (normal+subsampled), or 2pathway, or 2pathway+subsampled.
            list_id, list_labels: lists of strings of names of files
            segment_size=25: size of training segments
            channels: list of channels to include in the normal pathway
            subsample: bool. do you want to use subsampled pathway of DM
            sampling: sampling strategy, list of size nr.channels
            channels_sub: list of channels to use for subsampled pathway
            segment_size2: size of patches for second second input, should be odd. None if no second input used.
            channels2: which channels to use for second input, if in use."""

        #COMMENTS: I hardcode some DeepMed details, eg outsize=segment_size-16, subsampled_seg_size = seg_size+2*16-2,
        #           subsampling factor=3, ...
        # Also hardcoded are certain poem specifics (3d, expected shape of data, etc).
        self.list_IDs = list_IDs
        self.labels = list_labels
        self.sampling = sampling
        self.num_classes = num_classes
        self.subbatch = sum(sampling)
        self.segment = int((segment_size-1)/2)
        self.channels = channels
        self.outsize = int((segment_size-16)/2)   # x + N ->(:3)-> (x+2)/3+(N-2)/3 ->(-16)-> (x+2)/3-16+(N-2)/3 ->(*3)-> x+2+(N-2)/3*3-16*3 ==x-16
                                                                                           # (N-2)/3*3 = 2*16 - 2
        self.subsample = subsample
        self.subchannels = channels_sub
        self.subsegment = 27  #obv should be bigger than segment.
        #bigsegmentsize should be divisible by 3, as we downsample it. But also odd, to contain the center pixel.
        self.f = 3 #subsample factor. can be also input? But: bigsegment%%f=0, and bigsegm/f==odd.

        self.in2 = False if segment_size2==None else True
        self.in2seg = segment_size2 #here we assume this one is not the biggest of all used segments.
        self.in2ch = channels2

    def __len__(self):
        'total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, item):
        'get one sample: a subbatch of patches from 1 subject'
        #select subject
        subjdata = np.load(self.list_IDs[item], allow_pickle=True) 
        subj = subjdata['channels']
        dic = subjdata['organ_sizes'].item()
        label = np.load(self.labels[item])
        #get patches according to sampling strategy.
        #first let's correct sampling for the given subj:
        localSampling = self.sampling
        for org in range(self.num_classes):
            razlika = self.sampling[org]-dic[org]
            if 0<razlika:
                localSampling[org] = dic[org]
                localSampling[0] += razlika
        #now get center points, careful with where you are allowed to sample - try to use point as centre, if you cant, just sample 
        #somehting around it:
        s = label.shape
        #print(subj[0].shape)
        #print(s)
        bigsegment = self.segment
        if self.subsample:
            bigsegment = self.subsegment

  #      label_for_sampling = label[bigsegment:-bigsegment, bigsegment:-bigsegment, bigsegment:-bigsegment] 
        centres = []
        for org in range(self.num_classes):
  #          alla = np.column_stack(np.where(label_for_sampling==org))
            alla = np.column_stack(np.where(label==org))
            alla = alla[np.random.choice(alla.shape[0], localSampling[org], replace=False),...]
            centres.append(alla)

 #       centres = [ctr+bigsegment for ctr in centres if ctr]
        centres = np.concatenate(centres, axis=0)
        #now sample:
        patches =  []
        patchsub = []
        patches2 = []
        truths  =  []
        #Fat,Wat,Xdist,Ydist = subj[0], subj[1], subj[2], subj[3]
        newcentres = []
        for center in centres:
            starts = center-self.segment
            ends = center+self.segment+1
            below_start = [0 if st>=0 else -st for st in starts]
            over_ends = [0 if en<=0 else -en for en in ends-s]
            assert sum(np.multiply(over_ends, below_start))==0, (below_start, over_ends) #we expect to have problem max on one side of any dim
         #   print(f"old starts: {starts}, old ends:{ends}, over_ends: {over_ends}")
         #   if sum(over_ends+below_start)!=0:
         #       print(f"old starts: {starts}, old ends:{ends}, over_ends: {over_ends}")
         #       print(f"new starts: {starts+over_ends+below_start}, new ends: {ends+over_ends+below_start}")
            starts = starts+over_ends+below_start
            ends = ends+over_ends+below_start
         #   print(f"new starts: {starts}, new ends: {ends}")
            #get the new centre-point for subsampled inp, if needed:
            newcentres.append(starts+self.segment)
            
            newpatch = subj[self.channels, starts[0]:ends[0],
                                            starts[1]:ends[1],
                                            starts[2]:ends[2]]

            patches.append(torch.from_numpy(newpatch))
            truths.append(torch.from_numpy(label[starts[0]+8:ends[0]-8,
                                                starts[1] +8:ends[1]-8,
                                                starts[2] +8:ends[2]-8]))
        out = [torch.stack(patches)]

        if self.subsample:
            for center in newcentres:
                i, j, k = center
                padded = np.pad(subj, ((0,), (self.subsegment,), (self.subsegment,), (self.subsegment,)), 'symmetric')
                patchsub.append(torch.from_numpy(padded[self.subchannels, i:i + self.subsegment*2 + 1:self.f,
                                                                          j:j + self.subsegment*2 + 1:self.f,
                                                                          k:k + self.subsegment*2 + 1:self.f]))
                # axis 0 bcs in pytorch channels first. so for each patch we append channels x sgm x sgm x sgm array.
            out.append(torch.stack(patchsub))
        if self.in2:
            self.in2seg = int((self.in2seg-1)/2)
            for center in newcentres:
                i, j, k = center
                padded = np.pad(subj, ((0,), (self.in2seg,), (self.in2seg,), (self.in2seg,)), 'symmetric')
                patches2.append(torch.from_numpy(padded[self.in2ch, i:i + self.in2seg*2 + 1:self.f,
                                                                    j:j + self.in2seg*2 + 1:self.f,
                                                                    k:k + self.in2seg*2 + 1:self.f]))
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

