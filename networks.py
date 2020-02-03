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

class OnePathway(nn.Module):
  def __init__(self, in_channels, num_classes, dropoutrateCon=0.2, dropoutrateFC=0.5):
    super(OnePathway, self).__init__()
    self.conv1 = nn.Conv3d(in_channels, 30, 3)
    self.conv2 = nn.Conv3d(30, 30, 3)
    self.dropC1 = nn.Dropout3d(p=dropoutrateCon)
    self.conv3 = nn.Conv3d(30, 40, 3)
    self.conv4 = nn.Conv3d(40, 40, 3)
    self.dropC2 = nn.Dropout3d(p=dropoutrateCon)
    self.conv5 = nn.Conv3d(40, 40, 3)
    self.conv6 = nn.Conv3d(40, 40, 3)
    self.dropC3 = nn.Dropout3d(p=dropoutrateCon)
    self.conv7 = nn.Conv3d(40, 50, 3)
    self.conv8 = nn.Conv3d(50, 50, 3)
    self.dropC4 = nn.Dropout3d(p=dropoutrateCon)
    self.fcconv1 = nn.Conv3d(50, 150, 1)
    self.dropFC1 = nn.Dropout3d(p=dropoutrateFC)
    self.fcconv2 = nn.Conv3d(150, 150, 1)
    self.dropFC2 = nn.Dropout3d(p=dropoutrateFC)
    self.finalclass = nn.Conv3d(150, num_classes, 1)

  def forward(self, input):
    x = F.relu(self.conv1(input))
    x = F.relu(self.conv2(x))
    x = self.dropC1(x)
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.dropC2(x)
    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
    x = self.dropC3(x)
    x = F.relu(self.conv7(x))
    x = F.relu(self.conv8(x))
    x = self.dropC4(x)
    x = F.relu(self.fcconv1(x))
    x = self.dropFC1(x)
    x = F.relu(self.fcconv2(x))
    x = self.dropFC2(x)
    out = self.finalclass(x)
    return out

#two inputs
#class TwoInputsDM(nn.Module):
#  def __init__(self):
#    super(TwoInputsDM, self).__init__()
#    self.conv = nn.Conv2d( ... )  # set up your layer here
#    self.fc1 = nn.Linear( ... )  # set up first FC layer
#    self.fc2 = nn.Linear( ... )  # set up the other FC layer
#
#  def forward(self, input1, input2):
#    c = self.conv(input1)
#    f = self.fc1(input2)
#    # now we can reshape `c` and `f` to 2D and concat them
#    combined = torch.cat((c.view(c.size(0), -1),
#                          f.view(f.size(0), -1)), dim=1)
#    out = self.fc2(combined)
#    return out

