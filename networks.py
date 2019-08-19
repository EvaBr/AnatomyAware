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
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv3D, UpSampling3D, Concatenate, Reshape
import tensorflow as tf

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
