#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:39:54 2019

@author: eva
"""

import keras.backend as K
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

#################################### READING DATA #############################
def get_file_names(pot='/media/eva/Elements/backup-june2019'):
    dists = pot+'/Desktop/research/PROJEKT2\ -\ DeepLearning/distmaps/'
    fatwatmask = pot+'/Desktop/research/PROJEKT2\ -\ DeepLearning/procesiranDataset/POEM_segmentation_data_fatwat/converted/'
    labele = pot+'/Desktop/research/PROJEKT2\ -\ DeepLearning/procesiranDataset/POEM_segment_all/converted/'
    
    distx = glob.glob(dists+'*_x.nii'); distx.sort()
    disty = glob.glob(dists+'*_y.nii'); disty.sort()
    fats = glob.glob(fatwatmask+'*fat_content.nii'); fats.sort()
    wats = glob.glob(fatwatmask+'*wat_content.nii'); wats.sort()
    masks = glob.glob(fatwatmask+'*mask.nii'); masks.sort()
    labels = glob.glob(labele+'croppedSegm*'); labels.sort()
    
    list_ID = [[fats[i], wats[i], distx[i], disty[i], labels[i], masks[i]] for i in range(len(distx))]
    return list_ID


##################################### LOSSES & METRICS ########################
#weighted categ. crossentropy
    

#Dice loss
    


#more?
    






###################################### PLOTTING ###############################

def compareimages(GT, out, fati):
    offset=60
    maskedGT = np.ma.masked_where(GT == 0, GT)
    maskedOUT = np.ma.masked_where(out == 0, out)
    s=GT.shape
    k=s[2]//2
    grid = plt.GridSpec(8, 8, wspace=0.3, hspace=0.9)
    
    fig = plt.figure(figsize=(12,6)); 
    plt.subplot(grid[0:7,0:4])
    a = plt.imshow(fati[:,:,k], cmap='gray', extent=(0, 3*s[1], 0, s[0]))
    b = plt.imshow(maskedGT[:,:,k], cmap='jet', vmin=0, vmax=5, interpolation='none', alpha=0.7, extent=(0, 3*s[1], 0, s[0]))
    plt.subplot(grid[0:7, 4:])
    c = plt.imshow(fati[:,:,k], cmap='gray', extent=(0, 3*s[1], 0, s[0]))
    d = plt.imshow(maskedOUT[:,:,k], cmap='jet', vmin=0, vmax=5, interpolation='none', alpha=0.7, extent=(0, 3*s[1], 0, s[0]))

    slider = Slider(fig.add_subplot(grid[7,2:7]), 'Slice number: %i ' % k, 0+offset, s[2]-offset, valinit=0, valfmt='%i', 
                                    facecolor='g', edgecolor='w')
    slider.vline.set_color('green')
    def update(val):
        ind = int(slider.val)
        a.set_data(fati[:,:,ind])
        b.set_data(maskedGT[:,:,ind])
        c.set_data(fati[:,:,ind])
        d.set_data(maskedOUT[:,:,ind])
        fig.canvas.draw()

    slider.on_changed(update)

    plt.show()