#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:59:50 2019
The main things for deep medic implementation with additional spatial info.
Possibilities for running: normal deep med, with additional input channel
of distance maps, with mid-stage and late fusion of coordinates. 

@author: eva
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import PReLU as prelju
from functools import partial


#TODO: leaky relu, batchnorm, anything else form the orig.paper? # DONE.
#TODO: multiinput network!!

class OnePathway(nn.Module):
  def __init__(self, in_channels, num_classes, dropoutrateCon=0.2, dropoutrateFC=0.5, nonlin=nn.PReLU()):
    super(OnePathway, self).__init__()

    self.name = "OnePathway"

    self.dropoutrateCon = dropoutrateCon
    self.dropoutrateFC = dropoutrateFC
    self.nonlin = nonlin

    self.conv1 = nn.Conv3d(in_channels, 30, 3)
    self.convs = nn.ModuleList([])
    self.BNs = nn.ModuleList([])

    convols = []
    batchnorms = []
    nrfeat = 30
    for feats in [30,40,40,40,40,50,50]:
      convols.append(nn.Conv3d(nrfeat, feats, 3))
      batchnorms.append(nn.BatchNorm3d(nrfeat))
      nrfeat = feats
    self.convs.extend(convols)
    self.BNs.extend(batchnorms)

    self.BNfc1 = nn.BatchNorm3d(50)
    self.fcconv1 = nn.Conv3d(50, 150, 1)
    self.BNfc2 = nn.BatchNorm3d(150)
    self.fcconv2 = nn.Conv3d(150, 150, 1)
    self.lastBN = nn.BatchNorm3d(150)
    self.finalclass = nn.Conv3d(150, num_classes, 1)
    

  def forward(self, input):  # BN-ReLU-Conv + Dropout
    out = self.conv1(input)
    for cc, bn in zip(self.convs, self.BNs):
      out = bn(out)
      out = self.nonlin(out)
      out = cc(out)
      out = nn.Dropout3d(p=self.dropoutrateCon)(out)
    
    out = self.BNfc1(out)
    out = self.nonlin(out)
    out = self.fcconv1(out)
    out = nn.Dropout3d(p=self.dropoutrateFC)(out)
    out = self.BNfc2(out)
    out = self.nonlin(out)
    out = self.fcconv2(out)
    out = nn.Dropout3d(p=self.dropoutrateFC)(out)

    out = self.lastBN(out)
    out = self.nonlin(out)

    out = self.finalclass(x)
    return out


#two inputs
class DualPathway(nn.Module):
  def __init__(self, in_channels_orig, in_channels_subs, num_classes, dropoutrateCon=0.2, dropoutrateFC=0.5, nonlin=nn.PReLU()):
    super(DualPathway, self).__init__()
    # BN-ReLU-Conv + Dropout

    self.name = "DualPathway"

    self.dropoutrateCon = dropoutrateCon
    self.dropoutrateFC = dropoutrateFC

    self.conv1_p1 = nn.Conv3d(in_channels_orig, 30, 3)
    self.conv1_p2 = nn.Conv3d(in_channels_subs, 30, 3)
    self.con_p1 = nn.ModuleList([])
    self.con_p2 = nn.ModuleList([])
    self.BNs = nn.ModuleList([])

    batchnorms = []  #do we need separate Batchnorms for each of the pathways??
    convols_p1 = []
    convols_p2 = []
    nrfeat=30
    for feats in [30,40,40,40,40,50,50]:
      convols_p1.append(nn.Conv3d(nrfeat, feats, 3))
      convols_p2.append(nn.Conv3d(nrfeat, feats, 3))
      batchnorms.append(nn.BatchNorm3d(nrfeat))
      nrfeat = feats
    self.con_p1.extend(convols_p1)
    self.con_p2.extend(convols_p2)
    self.BNs.extend(batchnorms)

    self.jointBN = nn.BatchNorm3d(100)

    self.upsample = partial(F.interpolate, scale_factor=3, mode='nearest')

    self.fcconv1 = nn.Conv3d(100, 150, 1)
    self.BNfc1 = nn.BatchNorm3d(150)
    self.fcconv2 = nn.Conv3d(150, 150, 1)
    self.BNfc2 = nn.BatchNorm3d(150)
    self.finalclass = nn.Conv3d(150, num_classes, 1)

    self.nonlin = nonlin

  def forward(self, input1, input2): #(BN-ReLU-Conv-DropOut)
    input1 = self.conv1_p1(input1)
    input2 = self.conv1_p2(input2)

    for p1, p2, bn in zip(self.con_p1, self.con_p2, self.BNs):
      input1 = bn(input1)
      input1 = self.nonlin(input1)
      input1 = p1(input1)
      input1 = nn.Dropout3d(p=self.dropoutrateCon)(input1)

      input2 = bn(input2)
      input2 = self.nonlin(input2)
      input2 = p1(input2)
      input2 = nn.Dropout3d(p=self.dropoutrateCon)(input2)

    input2 = self.upsample(input2)

    out = torch.cat((input1, input2), dim=1)
    out = self.jointBN(out)  #Q: should we have the BatchNorm here?

    out = self.nonlin(out)
    out = self.fcconv1(out)
    out = nn.Dropout3d(p=self.dropoutrateFC)(out)
    
    out = self.BNfc1(out) #Q: Should we have batchnorm here?
    
    out = self.nonlin(out)
    out = self.fcconv2(out)
    out = nn.Dropout3d(p=self.dropoutrateFC)(out)

    out = self.BNfc2(out) #Q: Should we have batchnorm here?

    out = self.nonlin(out)
    out = self.finalclass(out)
   # out = nn.Dropout3d(p=self.dropoutrateFC)(out) #TODO: Q: do we really keep the final dropout???

    return out



class MultiPathway(nn.Module):
  def __init__(self, in_channels_orig, in_channels_subs, in_channels_add, join_at, join_to_orig, add_FM_sizes, num_classes=7, 
                    dropoutrateCon=0.2, dropoutrateFC=0.5, nonlin=nn.PReLU()):
    super(MultiPathway, self).__init__()
    #join_at = nr of layer at which to join (concatenate) the third pathway. numberign goes from 0-9; ie if join_at==0, both pathways are concat. just 
    #     after the first conv. If join_at==9, they're joined just before last (classifying) convolution.
    #join_to_orig = True if third path joined to orig pathway. Otherwise joined to subsampled path (if exists).
    #add_FM_sizes = list of nr of features per layer for third pathway. Convs are hardcoded to be 3x3 in all pathways.
    # (Aslo keep in mind that input patch size for third pathway needs to be large enough such convolution pipeline...at least 1 + 2*len(add_FM_sizes))
    #in_channels_add = nr of channels used in the thirdpathway
    #in_channels_subs = nr of channels used in subsampled pathway. If None or 0, no subsampled pathway built.
    # Additional pathway assumed mandatory, subsampled not mandatory.

    ns = ""
    self.using_subs = False
    self.join_at = join_at
    self.join_to_orig = join_to_orig
    if in_channels_subs: 
      ns = "_WithSubsampled"
      self.using_subs = True
    else:
      self.join_to_orig = True
    self.name = "MultiPathway"+ns

    if join_to_orig:
      expected_sizes = [23, 21, 19, 17, 15, 13, 11, 9, 9, 9]
    else:
      expected_sizes = [17, 15, 13, 11, 9, 7, 5, 3, 9, 9]
    newsize = expected_sizes[join_at] #the size the third pathway should be resampled to, so that we can concatenate it.

    self.dropoutrateCon = dropoutrateCon
    self.dropoutrateFC = dropoutrateFC
    self.nonlin = nonlin

    self.conv1_p1 = nn.Conv3d(in_channels_orig, 30, 3)
    self.conv1_p3 = nn.Conv3d(in_channels_add, add_FM_sizes[0], 3)
    if in_channels_subs:
      self.conv1_p2 = nn.Conv3d(in_channels_subs, 30, 3)
    self.con_p1 = nn.ModuleList([])
    self.con_p3 = nn.ModuleList([])
    self.con_p2 = nn.ModuleList([])

    self.BNs_p3 = nn.ModuleList([])
    self.BNs_p1 = nn.ModuleList([]) 
    self.BNs_p2 = nn.ModuleList([])  

    #original and possibly subsampled layer:
    deepmed_FM_sizes_orig = [30,40,40,40,40,50,50] #hardcoding for now.
    deepmed_FM_sizes_subs = [] #hardcoding for now.
    if in_channels_subs:
      deepmed_FM_sizes_subs = [30,40,40,40,40,50,50]
    if join_at<7:
      if join_to_orig:
        deepmed_FM_sizes_orig[join_at] += add_FM_sizes[-1] #where we join, we now have more FMs. 
      else: 
        deepmed_FM_sizes_subs[join_at] += add_FM_sizes[-1] #where we join, we now have more FMs. 
    
    #build orig path:
    nrfeat=30
    convols_p1 = []
    batchnorms_p1 = []
    for feats in deepmed_FM_sizes_orig:
      convols_p1.append(nn.Conv3d(nrfeat, feats, 3))
      batchnorms_p1.append(nn.BatchNorm3d(nrfeat))
      nrfeat = feats
    self.con_p1.extend(convols_p1)
    self.BNs_p1.extend(batchnorms_p1)
    #build subsampled path:
    nrfeat=30
    convols_p2 = []
    batchnorms_p2 = []
    for feats in deepmed_FM_sizes_subs:
      convols_p2.append(nn.Conv3d(nrfeat, feats, 3))
      batchnorms_p2.append(nn.BatchNorm3d(nrfeat))
      nrfeat = feats
    self.con_p2.extend(convols_p2)
    self.BNs_p2.extend(batchnorms_p2) #will stay empty if no subsamp pathway used.
    #build additional path:
    nrfeat=add_FM_sizes[0]
    convols_p3 = []
    batchnorms_p3 = []
    for feats in add_FM_sizes[1:]:
      convols_p3.append(nn.Conv3d(nrfeat, feats, 3))
      batchnorms_p3.append(nn.BatchNorm3d(nrfeat))
      nrfeat = feats
    self.con_p3.extend(convols_p3)
    self.BNs_p3.extend(batchnorms_p3)

    finals = [50, 150, 150]
    if join_at>=7:
      finals[join_at%7] += add_FM_sizes[-1]
    if using_subs:
      finals[0] += 50
    
    self.jointBN = nn.BatchNorm3d(finals[0])

    self.upsample_s = partial(F.interpolate, scale_factor=3, mode='nearest') #upsampling for subsamled path, if used
    self.upsample_a = partial(F.interpolate, size=(newsize, newsize, newsize)) #upsampling for the additional path

    self.fcconv1 = nn.Conv3d(finals[0], 150, 1)
    self.BNfc1 = nn.BatchNorm3d(finals[1])
    self.fcconv2 = nn.Conv3d(finals[1], 150, 1)
    self.BNfc2 = nn.BatchNorm3d(finals[2])
    self.finalclass = nn.Conv3d(finals[2], num_classes, 1)

  def forward(self, input1, input2, input3=None):
    input1 = self.conv1_p1(input1)
    if input3:
      input3 = self.conv1_p3(input3)
      input2 = self.conv1_p2(input2)
      assert self.using_subs
    else:
      input3 = self.conv1_p3(input2)
      assert self.using_subs==False

    for p, bn in zip(self.con_p3, self.BNs_p3): #will be empty if input2 not used.
      input3 = bn(input3)
      input3 = self.nonlin(input3)
      input3 = p(input3)
      input3 = nn.Dropout3d(p=self.dropoutrateCon)(input3)

    for p, bn in zip(self.con_p1, self.BNs_p1):
      input1 = bn(input1)
      input1 = self.nonlin(input1)
      input1 = p(input1)
      input1 = nn.Dropout3d(p=self.dropoutrateCon)(input1)

    for p, bn in zip(self.con_p2, self.BNs_p2): #will be empty if input2 not used.
      input2 = bn(input2)
      input2 = self.nonlin(input2)
      input2 = p(input2)
      input2 = nn.Dropout3d(p=self.dropoutrateCon)(input2)


    if self.using_subs: #if we are using subsampled thing
      input2 = self.upsample_s(input2)

    out = torch.cat((input1, input2), dim=1)
    out = self.jointBN(out)  #Q: should we have the BatchNorm here?

    out = self.nonlin(out)
    out = self.fcconv1(out)
    out = nn.Dropout3d(p=self.dropoutrateFC)(out)
    
    out = self.BNfc1(out) #Q: Should we have batchnorm here?
    
    out = self.nonlin(out)
    out = self.fcconv2(out)
    out = nn.Dropout3d(p=self.dropoutrateFC)(out)

    out = self.BNfc2(out) #Q: Should we have batchnorm here?

    out = self.nonlin(out)
    out = self.finalclass(out)
   # out = nn.Dropout3d(p=self.dropoutrateFC)(out) #TODO: Q: do we really keep the final dropout???
#TODO: NEED TO ADD CONCATENATION OF input3!
    return out







#more portable, but not working (gives seg fault):

class DenseLayer(nn.Module):
  def __init__(
          self,
          in_channels,
          out_channels,
          batch_norm="batchnorm",
          dropout=0.2,
          kernel_size=3,
          act=nn.PReLU(),
    ):
    super().__init__()
    self.dropout = dropout
    self.has_bn = batch_norm is not None
    # Standard Tiramisu Layer (BN-ReLU-Conv-DropOut)
    if batch_norm == "batchnorm":
      self.add_module("batchnorm", nn.BatchNorm3d(in_channels))
    self.add_module("act", act)
    self.add_module(
      "conv",
      nn.Conv3d(
        in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=True
      ),
    )

  def forward(self, x):
    x = self.batchnorm(x)
    x = self.act(x)
    x = self.conv(x)
    if self.dropout and self.dropout != 0.0:
      x = F.dropout(x, p=self.dropout, training=self.training)
    return x




class DualPathwayNico(nn.Module):
  def __init__(self, in_channels_orig, in_channels_subs, num_classes, dropoutrateCon=0.2, dropoutrateFC=0.5, nonlin=nn.PReLU()):
    super(DualPathwayNico, self).__init__()

    self.name = "DualPathwayNico"

    self.init_conv1 = nn.Conv3d(in_channels_orig, 30, 3)
    self.init_conv2 = nn.Conv3d(in_channels_subs, 30, 3)

    self.path1 = nn.ModuleList([])
    self.path2 = nn.ModuleList([])
    path1, path2 = [], []
    in_channels = 30
    out_channels = 30
    for i in range(8):
      path1.append(DenseLayer(in_channels, out_channels, dropout=dropoutrateCon, act=nonlin))
      path2.append(DenseLayer(in_channels, out_channels, dropout=dropoutrateCon, act=nonlin))
      in_channels = out_channels
      out_channels += 10
    self.path1.extend(path1)
    self.path2.extend(path2)

    self.upsample = partial(F.interpolate, scale_factor=3, mode='nearest')

    self.final = nn.ModuleList([
      DenseLayer(100, 150, kernel_size=1, dropout=dropoutrateFC, act=nonlin),
      DenseLayer(150, 150, kernel_size=1, dropout=dropoutrateFC, act=nonlin),
      DenseLayer(150, num_classes, kernel_size=1, dropout=dropoutrateFC, act=nonlin)
    ])

  def forward(self, input1, input2):
    input1 = self.init_conv1(input1)
    input2 = self.init_conv2(input2)

    for i in range(len(self.path1)):
        input1 = self.path1[i](input1)
        input2 = self.path2[i](input2)
    input2 = self.upsample(input2)

    x = torch.cat((input1, input2), dim=1)

    for i in range(len(self.final)):
      x = self.final[i](x)
    
    return x




