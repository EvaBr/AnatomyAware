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
    self.conv1 = nn.Conv3d(in_channels, 30, 3, padding=0)
    self.conv2 = nn.Conv3d(30, 30, 3)
    self.conv3 = nn.Conv3d(30, 40, 3, padding=0)
    self.conv4 = nn.Conv3d(40, 40, 3)
    self.conv5 = nn.Conv3d(40, 40, 3, padding=0)
    self.conv6 = nn.Conv3d(40, 40, 3)
    self.conv7 = nn.Conv3d(40, 50, 3, padding=0)
    self.conv8 = nn.Conv3d(50, 50, 3)
    self.fcconv1 = nn.Conv3d(50, 150, 1)
    self.fcconv2 = nn.Conv3d(150, 150, 1)
    self.finalclass = nn.Conv3d(150, num_classes, 1)

  def forward(self, input):
    x = self.conv1(input)
    x = self.conv2(x)
    x = self.conv3(x)
    
    # now we can reshape `c` and `f` to 2D and concat them
    combined = torch.cat((c.view(c.size(0), -1),
                          f.view(f.size(0), -1)), dim=1)
    out = self.fc2(combined)
    return out

#two inputs
class TwoInputsDM(nn.Module):
  def __init__(self):
    super(TwoInputsDM, self).__init__()
    self.conv = nn.Conv2d( ... )  # set up your layer here
    self.fc1 = nn.Linear( ... )  # set up first FC layer
    self.fc2 = nn.Linear( ... )  # set up the other FC layer

  def forward(self, input1, input2):
    c = self.conv(input1)
    f = self.fc1(input2)
    # now we can reshape `c` and `f` to 2D and concat them
    combined = torch.cat((c.view(c.size(0), -1),
                          f.view(f.size(0), -1)), dim=1)
    out = self.fc2(combined)
    return out


#original deepmed
def orig_deepmed(num_classes):
    # define two sets of inputs
    inputA = Input(shape=(25,25,25,2)) #2channels
    inputB = Input(shape=(19,19,19,4)) #4channels
 
    # the first branch operates on the first input
    x = Conv3D(30, (3,3,3), activation="relu")(inputA)
    x = Conv3D(30, (3,3,3), activation="relu")(x)

    x = Dropout(rate=0.2)(x)
    
    x = Conv3D(40, (3,3,3), activation="relu")(x)
    x = Conv3D(40, (3,3,3), activation="relu")(x)
    
    x = Dropout(rate=0.2)(x)
    
    x = Conv3D(40, (3,3,3), activation="relu")(x)
    x = Conv3D(40, (3,3,3), activation="relu")(x)

    x = Dropout(rate=0.2)(x)
    
    x = Conv3D(50, (3,3,3), activation="relu")(x)
    x = Conv3D(50, (3,3,3), activation="relu")(x)
    
    x = Model(inputs=inputA, outputs=x)
    
    
    # the second branch opreates on the second input
    y =  Conv3D(30, (3,3,3), activation="relu")(inputB)
    y = Conv3D(30, (3,3,3), activation="relu")(y)

    y = Dropout(rate=0.2)(y)
    
    y = Conv3D(40, (3,3,3), activation="relu")(y)
    y = Conv3D(40, (3,3,3), activation="relu")(y)
 
    y = Dropout(rate=0.2)(y)
       
    y = Conv3D(40, (3,3,3), activation="relu")(y)
    y = Conv3D(40, (3,3,3), activation="relu")(y)
    
    y = Dropout(rate=0.2)(y)
    
    y = Conv3D(50, (3,3,3), activation="relu")(y)
    y = Conv3D(50, (3,3,3), activation="relu")(y)
    
    y = UpSampling3D(size=(3, 3, 3))(y)
    y = Model(inputs=inputB, outputs=y)
    
    # combine the output of the two branches
    combined = Concatenate(axis=-1)([x.output, y.output])
    
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Conv3D(150, (1,1,1), activation="relu")(combined)
    z = Dropout(rate=0.5)(z)
    z = Conv3D(150, (1,1,1), activation="relu")(z)
    z = Dropout(rate=0.5)(z)
    z = Conv3D(num_classes,(1,1,1), activation="softmax")(z) 
    # our model will accept the inputs of the two branches and
    # then output a single value
    #LOOOL WHY DAFUQ DOES THIS NOT WORK::: z = keras.backend.argmax(z, -1)
    z = Reshape((9*9*9, num_classes))(z)
    model = Model(inputs=[inputA, inputB], outputs=z)

    return model    


#todo: daj to v funkcijo, DONE
#probi zagnat. DONE
#Dodaj batch norm,  DONE: BREZ!
#Dodaj dorpout, DONE
#poglej kako ma  kamnitsas (vrstni red dropouta in activacij, also mogoce aktivacije zunaj conv3d dodadj?
#also a je classif. layer ok da je conv3d?): 
  # ---------------- Order of what is applied -----------------
  #  Input -> [ BatchNorm OR bias] -> NonLin -> DropOut -> Pooling --> Conv ] 
  # (ala He et al "Identity Mappings in Deep Residual Networks" 2016)
  # -----------------------------------------------------------



def late_fusion(num_classes):
    # define two sets of inputs
    inputA = Input(shape=(25,25,25,2)) #2channels
    inputB = Input(shape=(19,19,19,2)) #4channels
    inputC = Input(shape=(9, 9, 9, 2)) #2distance maps
 
    # the first branch operates on the first input
    x = Conv3D(30, (3,3,3), activation="relu")(inputA)
    x = Conv3D(30, (3,3,3), activation="relu")(x)

    x = Dropout(rate=0.2)(x)
    
    x = Conv3D(40, (3,3,3), activation="relu")(x)
    x = Conv3D(40, (3,3,3), activation="relu")(x)
    
    x = Dropout(rate=0.2)(x)
    
    x = Conv3D(40, (3,3,3), activation="relu")(x)
    x = Conv3D(40, (3,3,3), activation="relu")(x)

    x = Dropout(rate=0.2)(x)
    
    x = Conv3D(50, (3,3,3), activation="relu")(x)
    x = Conv3D(50, (3,3,3), activation="relu")(x)
    
    x = Model(inputs=inputA, outputs=x)
    
    
    # the second branch opreates on the second input
    y =  Conv3D(30, (3,3,3), activation="relu")(inputB)
    y = Conv3D(30, (3,3,3), activation="relu")(y)

    y = Dropout(rate=0.2)(y)
    
    y = Conv3D(40, (3,3,3), activation="relu")(y)
    y = Conv3D(40, (3,3,3), activation="relu")(y)
 
    y = Dropout(rate=0.2)(y)
       
    y = Conv3D(40, (3,3,3), activation="relu")(y)
    y = Conv3D(40, (3,3,3), activation="relu")(y)
    
    y = Dropout(rate=0.2)(y)
    
    y = Conv3D(50, (3,3,3), activation="relu")(y)
    y = Conv3D(50, (3,3,3), activation="relu")(y)
    
    y = UpSampling3D(size=(3, 3, 3))(y)
    y = Model(inputs=inputB, outputs=y)
    
    # combine the output of the two branches
    combined = Concatenate(axis=-1)([x.output, y.output, inputC])
    
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Conv3D(150, (1,1,1), activation="relu")(combined)
    z = Dropout(rate=0.5)(z)
    z = Conv3D(150, (1,1,1), activation="relu")(z)
    z = Dropout(rate=0.5)(z)
    z = Conv3D(num_classes,(1,1,1), activation="softmax")(z) 
    # our model will accept the inputs of the two branches and
    # then output a single value
    #LOOOL WHY DAFUQ DOES THIS NOT WORK::: z = keras.backend.argmax(z, -1)
    z = Reshape((9*9*9, num_classes))(z)
    model = Model(inputs=[inputA, inputB, inputC], outputs=z)

    return model    




def mid_fusion(num_classes):
    # define two sets of inputs
    inputA = Input(shape=(25,25,25,2)) #2channels
    inputB = Input(shape=(19,19,19,2)) #4channels
    inputC = Input(shape=(11,11,11,2)) #2distance maps
 
    # the first branch operates on the first input
    x = Conv3D(30, (3,3,3), activation="relu")(inputA)
    x = Conv3D(30, (3,3,3), activation="relu")(x)

    x = Dropout(rate=0.2)(x)
    
    x = Conv3D(40, (3,3,3), activation="relu")(x)
    x = Conv3D(40, (3,3,3), activation="relu")(x)
    
    x = Dropout(rate=0.2)(x)
    
    x = Conv3D(40, (3,3,3), activation="relu")(x)
    x = Conv3D(40, (3,3,3), activation="relu")(x)

    x = Dropout(rate=0.2)(x)
    
    x = Conv3D(50, (3,3,3), activation="relu")(x)
    x = Conv3D(50, (3,3,3), activation="relu")(x)
    
    x = Model(inputs=inputA, outputs=x)
    
    
    # the second branch opreates on the second input
    y =  Conv3D(30, (3,3,3), activation="relu")(inputB)
    y = Conv3D(30, (3,3,3), activation="relu")(y)

    y = Dropout(rate=0.2)(y)
    
    y = Conv3D(40, (3,3,3), activation="relu")(y)
    y = Conv3D(40, (3,3,3), activation="relu")(y)
 
    y = Dropout(rate=0.2)(y)
    
    #fusion in middle of lower pathway:
    y = Concatenate(axis=-1)([y, inputC])
    
    y = Conv3D(40, (3,3,3), activation="relu")(y)
    y = Conv3D(40, (3,3,3), activation="relu")(y)
    
    y = Dropout(rate=0.2)(y)
    
    y = Conv3D(50, (3,3,3), activation="relu")(y)
    y = Conv3D(50, (3,3,3), activation="relu")(y)
    
    y = UpSampling3D(size=(3, 3, 3))(y)
    y = Model(inputs=inputB, outputs=y)
    
    # combine the output of the two branches
    combined = Concatenate(axis=-1)([x.output, y.output])
    
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Conv3D(150, (1,1,1), activation="relu")(combined)
    z = Dropout(rate=0.5)(z)
    z = Conv3D(150, (1,1,1), activation="relu")(z)
    z = Dropout(rate=0.5)(z)
    z = Conv3D(num_classes,(1,1,1), activation="softmax")(z) 
    # our model will accept the inputs of the two branches and
    # then output a single value
    #LOOOL WHY DAFUQ DOES THIS NOT WORK::: z = keras.backend.argmax(z, -1)
    z = Reshape((9*9*9, num_classes))(z)
    model = Model(inputs=[inputA, inputB, inputC], outputs=z)

    return model    
