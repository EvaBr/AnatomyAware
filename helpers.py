#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:39:54 2019

@author: eva
"""

import glob
import numpy as np
import nibabel as nib
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

#################################### READING DATA #############################
def get_POEM_files(pot='/media/eva/Elements/backup-june2019'):
    dists = pot+'/Desktop/research/PROJEKT2\ -\ DeepLearning/distmaps/'
    fatwatmask = pot+'/Desktop/research/PROJEKT2\ -\ DeepLearning/procesiranDataset/POEM_segmentation_data_fatwat/converted/'
    labele = pot+'/Desktop/research/PROJEKT2\ -\ DeepLearning/procesiranDataset/POEM_segment_all/converted/'
    
    distx = glob.glob(dists+'*_x.nii'); distx.sort()
    disty = glob.glob(dists+'*_y.nii'); disty.sort()
    fats = glob.glob(fatwatmask+'*fat_content.nii'); fats.sort()
    wats = glob.glob(fatwatmask+'*wat_content.nii'); wats.sort()
    masks = glob.glob(pot + '*mask.nii'); masks.sort()
    labels = glob.glob(labele+'croppedSegm*'); labels.sort()

    #list_ID = np.array([[fats[i], wats[i], distx[i], disty[i], labels[i], masks[i]] for i in range(len(distx))])
    list_ID = np.array([[fats[i], wats[i], distx[i], disty[i], get_label_and_info(labels[i], masks[i])] for i in range(len(distx))])

    rand_perm = np.random.permutation(len(list_ID))
    tr, va, te = rand_perm[0:35], rand_perm[35:45], rand_perm[45:]
    train, validate, test = list_ID[tr], list_ID[va], list_ID[te]

    with open('datasetTRAIN.pickle', 'wb') as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('datasetVALIDATE.pickle', 'wb') as handle:
        pickle.dump(validate, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('datasetTEST.pickle', 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train, validate, test

def get_label_and_info(label, mask):
    #pot = '/media/eva/Elements/backup-june2019/Desktop/research/PROJEKT2\ -\ DeepLearning/procesiranDataset/POEM_segmentation_data_fatwat/converted/'
    #potlab = '/media/eva/Elements/backup-june2019/Desktop/research/PROJEKT2\ -\ DeepLearning/procesiranDataset/POEM_segment_all/converted/'
    #masks = glob.glob(pot + '*mask.nii')
    #masks.sort()
    #labels = glob.glob(potlab + 'croppedSegm*')
    #labels.sort()
    #label_sizes = []
    #for i in range(len(labels)):
        #maska = nib.load(masks[i]).get_fdata()
        #labela = nib.load(labels[i]).get_fdata()
        #label_sizes.append({5:sum(labela==5), 4:sum(labela==4), 3:sum(labela==3), 2:sum(labela==2), 1:sum(labela==1), 0:sum(maska>0 & labela==0)})
    labela = nib.load(label).get_fdata()
    maska = nib.load(mask).get_fdata()
    return (labela - 1*(maska==0)), {5: sum(labela == 5), 4: sum(labela == 4), 3: sum(labela == 3), 2: sum(labela == 2), 1: sum(labela == 1),
     0: sum(maska > 0 & labela == 0)}

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
    
    fig = plt.figure(figsize=(12, 6));
    plt.subplot(grid[0:7, 0:4])
    a = plt.imshow(fati[:, :, k], cmap='gray', extent=(0, 3*s[1], 0, s[0]))
    b = plt.imshow(maskedGT[:, :, k], cmap='jet', vmin=0, vmax=5, interpolation='none', alpha=0.7, extent=(0, 3*s[1], 0, s[0]))
    plt.subplot(grid[0:7, 4:])
    c = plt.imshow(fati[:, :, k], cmap='gray', extent=(0, 3*s[1], 0, s[0]))
    d = plt.imshow(maskedOUT[:, :, k], cmap='jet', vmin=0, vmax=5, interpolation='none', alpha=0.7, extent=(0, 3*s[1], 0, s[0]))

    slider = Slider(fig.add_subplot(grid[7, 2:7]), 'Slice number: %i ' % k, 0+offset, s[2]-offset, valinit=0, valfmt='%i',
                                    facecolor='g', edgecolor='w')
    slider.vline.set_color('green')
    def update(val):
        ind = int(slider.val)
        a.set_data(fati[:, :, ind])
        b.set_data(maskedGT[:, :, ind])
        c.set_data(fati[:, :, ind])
        d.set_data(maskedOUT[:, :, ind])
        fig.canvas.draw()

    slider.on_changed(update)

    plt.show()


