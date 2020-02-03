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
import torch
import torch.nn as nn
import torch.nn.functional as F

#################################### READING DATA #############################
def get_POEM_files(pot='/media/eva/Elements/backup-june2019'):
    dists = pot+'/Desktop/research/PROJEKT2-DeepLearning/distmaps/'
    fatwatmask = pot+'/Desktop/research/PROJEKT2-DeepLearning/procesiranDataset/POEM_segmentation_data_fatwat/converted/'
    labele = pot+'/Desktop/research/PROJEKT2-DeepLearning/procesiranDataset/POEM_segment_all/converted/'

    distx = glob.glob(dists+'*_x.nii'); distx.sort()
    disty = glob.glob(dists+'*_y.nii'); disty.sort()
    fats = glob.glob(fatwatmask+'*fat_content.nii'); fats.sort()
    wats = glob.glob(fatwatmask+'*wat_content.nii'); wats.sort()
    masks = glob.glob(fatwatmask + '*mask.nii'); masks.sort()
    labels = glob.glob(labele+'croppedSegm*'); labels.sort()
    #print(len(labels))
    #print(len(wats))
    #print(len(masks))

    #list_ID = np.array([[fats[i], wats[i], distx[i], disty[i], labels[i], masks[i]] for i in range(len(distx))])
    IDji = []
    for i in range(len(distx)):
        newlabel, slovar = get_label_and_info(labels[i], masks[i])
        IDji.append([fats[i], wats[i], distx[i], disty[i], newlabel, slovar])
    list_ID = np.array(IDji)

    rand_perm = np.random.permutation(len(list_ID))
    tr, va, te = rand_perm[0:30], rand_perm[30:40], rand_perm[40:]
    train, validate, test = list_ID[tr], list_ID[va], list_ID[te]

    with open('datasetTRAIN.pickle', 'wb') as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('datasetVALIDATE.pickle', 'wb') as handle:
        pickle.dump(validate, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('datasetTEST.pickle', 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train, validate, test

def get_label_and_info(label, mask):
    labela = nib.load(label).get_fdata()
    maska = nib.load(mask).get_fdata()
    return (labela - 1*(maska==0)), {5: np.sum(labela == 5), 4: np.sum(labela == 4), 3: np.sum(labela == 3), 2: np.sum(labela == 2), 1: np.sum(labela == 1),
     0: np.sum((maska > 0) & (labela == 0))}


def createDatasets(filename='AnatomyAware/datasetTRAIN.pickle', folder='AnatomyAware/TRAINdata'):
    with open(filename, 'rb') as handle:
        subjekti = pickle.load(handle)

    i=0
    for subj in subjekti:
      #  print(i)
        fat = nib.load(subj[0]).get_fdata()
      #  print(fat.shape)
        wat = nib.load(subj[1]).get_fdata()
        dix = nib.load(subj[2]).get_fdata()
        diy = nib.load(subj[3]).get_fdata()
      #  print(diy.shape)
        #namesto subjekti[4] bi lahko uporabljal labela-(maska==0), ce neb mel ze naprej shranjen.
        bckg = np.argwhere(subj[4]==0)
        org1 = np.argwhere(subj[4]==1)
        org2 = np.argwhere(subj[4]==2)
        org3 = np.argwhere(subj[4]==3)
        org4 = np.argwhere(subj[4]==4)
        org5 = np.argwhere(subj[4]==5)
        sezn = {0: len(bckg), 1: len(org1), 2: len(org2), 3: len(org3), 4: len(org4), 5: len(org5)}
        lab = subj[4]; lab[lab<0]=0
      #  print(lab.shape)
        i=i+1

        tmp = np.array([fat, wat, dix, diy, bckg, org1, org2, org3, org4, org5, sezn])
        with open(folder+'/subj{0}.pickle'.format(i), 'wb') as handle:
            pickle.dump(tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(folder+'/label{0}.pickle'.format(i), 'wb') as handle:
            pickle.dump(lab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #print('sezn: ', sezn, ' od prej: ', subj[5])


##############################  PROCESSING SEGMENTED DATA #####################
def glue_patches(patchi, placeringar): #glues, also performs nn.Softmax...
    l = int(np.sqrt(len(patchi)))
    s = patchi[0].shape # 6 x 256 x _ x 256
    print(s)
    segmented = np.empty(shape=(l*s[1], s[2], l*s[3]))
    for patch, place in zip(patchi, placeringar):
        patch = nn.Softmax(dim=0)(patch) #mogoce se zmeraj dim=1? odvisno, ce pustis dummy batch dimension
        x,y = place
        print(segmented[x*s[0]:(x+1)*s[0], :, y*s[2]:(y+1)*s[2]].shape, patch.data.numpy().shape)
        segmented[x*s[0]:(x+1)*s[0], :, y*s[2]:(y+1)*s[2]] = patch.data.numpy()

    return segmented


##################################### LOSSES & METRICS ########################
#weighted categ. crossentropy
    

#Dice loss
def dice_coeff(pred, target, nb_classes, weights):
    smooth = 0.0001
    #print(target.shape) #(12, 9,9,9)
    target = F.one_hot(target.long(), num_classes=nb_classes).permute(0, 4, 1, 2, 3).contiguous()
    #print(target.shape) #(12, 6, 9,9,9)
    weights = torch.from_numpy(weights)

    #now target is also of size BxCxHxWxL, with entries 1/0
#    num = pred.size(0)
#    m1 = pred.contiguous().view(num, -1)  # Flatten
#    m2 = target.contiguous().view(num, -1)  #torch.from_numpy( Flatten
#    intersection = torch.sum(m1 * m2.float())
#
#    return (2. * intersection + smooth) / (torch.sum(m1) + torch.sum(m2.float()) + smooth)
#    #We return shape Bx1: cause when you test, B=nr subj, so you don't want average Dice there....

    weights = weights.float().view(1, nb_classes, 1, 1, 1) #add dummy dimensions to broadcast when multiplying
    dims = tuple(range(1, target.ndimension())) #leave out batch-dim
    intersection = torch.sum(pred * target * weights, dims) #weighted sum over all classes
    cardinality = torch.sum((pred + target) * weights, dims) #weighted sum over all classes
    #print(dims)
    return (2. * intersection) / (cardinality + smooth)

def dice_coeff_per_class(pred, target, nb_classes):
    smooth = 0.0001
    # print(target.shape) #(12, 9,9,9)
    target = F.one_hot(target.long(), num_classes=nb_classes).permute(0, 4, 1, 2, 3).contiguous()
    dims = tuple(range(2, target.ndimension())) #leave out batch-dim and class-dim
    intersection = torch.sum(pred * target, dims) #weighted sum over all classes
    cardinality = torch.sum(pred + target, dims) #weighted sum over all classes
    #print(dims)
    return (2. * intersection) / (cardinality + smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, nb_classes=6, weight=np.array([1,1,1,1,1,1]), size_average=True):
        super(SoftDiceLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.weights = weight
        self.classes = nb_classes

    def forward(self, logits, targets):
        probs = self.softmax(logits)
        num = targets.size(0)  # Number of batches

        score = dice_coeff(probs, targets, self.classes, self.weights)
        score = 1 - score.sum() / num  # .sum je za sumacijo preko batchev...
        return score


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


