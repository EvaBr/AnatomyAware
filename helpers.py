#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:39:54 2019

@author: eva
"""

import glob
import natsort
import numpy as np
import nibabel as nib
import pickle
#import scipy.misc
import imageio
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial


tqdm_ = partial(tqdm, ncols=100,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

#################################### READING DATA #############################
def get_POEM_files(pot='/media/eva/Elements/backup-june2019'):
    outpath = '/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Data/'

    dists = '/home/eva/Desktop/research/PROJEKT2-DeepLearning/distmaps/'
    fatwatmask = '/home/eva/Desktop/research/PROJEKT2-DeepLearning/procesiranDataset/POEM_segmentation_data_fatwat/converted/'
    labele = '/home/eva/Desktop/research/PROJEKT2-DeepLearning/procesiranDataset/POEM_segment_all/converted/'

    distx = glob.glob(dists+'*_x.nii'); distx.sort()
    disty = glob.glob(dists+'*_y.nii'); disty.sort()
    fats = glob.glob(fatwatmask+'*fat_content.nii'); fats.sort()
    wats = glob.glob(fatwatmask+'*wat_content.nii'); wats.sort()
    masks = glob.glob(fatwatmask + '*mask.nii'); masks.sort()
    labels = glob.glob(labele+'CroppedSegmNew*'); labels.sort()

    #rabimo shranit := np.array([fats[i], wats[i], distx[i], disty[i]]), dict, labels[i] for i in range(len(distx))])
    rand_perm = np.random.permutation(len(distx))
    #tr, va, te = rand_perm[0:30], rand_perm[30:40], rand_perm[40:]
    for subjekt in range(len(distx)): #for every subject
        i = rand_perm[subjekt]

        newlabel, slovar = get_label_and_info(labels[i], masks[i])
        fat = nib.load(fats[i]).get_fdata()
        wat = nib.load(wats[i]).get_fdata()
        dix = nib.load(distx[i]).get_fdata()
        diy = nib.load(disty[i]).get_fdata()

        subj = np.stack([fat, wat, dix, diy], axis=0)
        #now save the object data in appropriate folder:
        if subjekt<30: 
            name = outpath + 'TRAINdata/'
        elif subjekt<40:
            name = outpath + 'VALdata/'
        else:
            name = outpath + 'TESTdata/'
        np.savez(name + f'subj_{i}', channels = subj, organ_sizes = slovar, allow_pickle=False)
        np.save(name + f'label_{i}', newlabel, allow_pickle=False)
    print(f"done. example label shape: {newlabel.shape}, and channels: {subj.shape}, and dictionary: {slovar}")

    return "Preparing POEM files done."



def get_label_and_info(label, mask):
    labela = nib.load(label).get_fdata()
    maska = nib.load(mask).get_fdata()
    return (labela - 1*(maska==0)), {6: np.sum(labela == 6), 5: np.sum(labela == 5), 4: np.sum(labela == 4), 
                                    3: np.sum(labela == 3), 2: np.sum(labela == 2), 1: np.sum(labela == 1),
                                    0: np.sum((maska > 0) & (labela == 0))}






##############################  PROCESSING SEGMENTED DATA #####################
def cut_patches(subj_list, patchsize, overlap, channels=4, outpath="", subsampledinput=False):
    #cut each img in subj_list into patches of size patchsize with given overlap (16).
    #if subsampledinput, cut also larger patches (size should be 3*(patchsize-6) for deepmed)
    # and downsample them, for second input.
    #outpath is where the cut imges will be saved, channels =1-4 is how many channels we use. First two are
    # fat and wat img, second two are x- and y-dist map.

    #OBS patchsize needs to be 1 modulo 3 !!!
    if patchsize%3!=1:
        raise ValueError('Patch size not equal to 1 modulo 3! Try again.')
    step = patchsize-overlap
    bigpatch = patchsize + 2*overlap - 2
    pad = (bigpatch-patchsize)//2 #==15
    out_list = []
    for sub in subj_list:
        subj = np.load(sub)
        subj = subj['channels']
        nr = re.findall(r'.*subj_([0-9]*)\.npz', sub)

        # now cut it to appropriate pieces
        s = subj.shape[1:]  # (256, x, 256)
        for i in range(s[0]//step+1): #cut so many pieces from left, and then one from right end, in case x,y,z not divisible with step.
            i_tmp = np.minimum(patchsize + (i + 1) * step, s[0])
            for j in range(s[1]//step+1):
                j_tmp = np.minimum(patchsize + (j + 1) * step, s[1])
                for k in range(s[2]//step+1):
                    k_tmp = np.minimum(patchsize + (k + 1) * step, s[2])
                    patch = subj[:, i_tmp-patchsize:i_tmp, j_tmp-patchsize:j_tmp, k_tmp-patchsize:k_tmp]
                 #   print("cut out a patch of size ", patch.shape)
                    nm = f"{outpath}/subj_{nr[0]}_{channels}_{i_tmp}_{j_tmp}_{k_tmp}.npy"
                    np.save(nm, patch)
                    out_list.append(nm)
        if subsampledinput:
            subj = np.pad(subj, ((0,0), (pad,pad), (pad,pad), (pad,pad)))
            for i in range(s[0] // step + 1):  # cut so many pieces from left, and then one from right end, in case x,y,z not divisible with step.
                i_tmp = np.minimum(patchsize + (i + 1) * step, s[0])
                for j in range(s[1] // step + 1):
                    j_tmp = np.minimum(patchsize + (j + 1) * step, s[1])
                    for k in range(s[2] // step + 1):
                        k_tmp = np.minimum(patchsize + (k + 1) * step, s[2])
                        patch = subj[:, i_tmp-patchsize:i_tmp + 2*pad:3, j_tmp-patchsize:j_tmp + 2*pad:3, k_tmp-patchsize:k_tmp + 2*pad:3]
                        nm = f"{outpath}/subj_{nr[0]}_{channels}_{i_tmp}_{j_tmp}_{k_tmp}_subsmpl.npy"
                        np.save(nm, patch)
    return out_list

def glue_patches(nr, path, patchsize, overlap, nb_classes=6): #glues, also performs nn.Softmax and saves png, returns numpy one one-hot
    patches = glob.glob(path+'subj_'+nr+'_*_OUT*')
    patches = natsort.natsorted(patches)
    sajz = np.array([nb_classes]+list(re.findall(r".*_([0-9]+)_([0-9]+)_([0-9]+)", patches[-1])[0]), np.int16)
    out_img = np.zeros(sajz)
    step = patchsize-2*overlap
    for pch in patches:
        s = np.array(re.findall(r".*_([0-9]+)_([0-9]+)_([0-9]+)", pch)[0], np.int16)
        out_img[:, s[0]-step:s[0], s[1]-step:s[1], s[2]-step:s[2]] = np.load(pch)
    one_hot_whole_subj = out_img[:, overlap*2:, overlap*2:, overlap*2:]
    #save a png:
    to_save = np.argmax(one_hot_whole_subj, axis=0)
    #imageio.imwrite(f'{path}out{nr}.jpg', to_save)
    np.save(f'{path}out{nr}.npy',to_save)
    return one_hot_whole_subj[np.newaxis,...]



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
    intersection = torch.sum(pred * target, dims)
    cardinality = torch.sum(pred + target, dims)
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
    'deprecated'
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


def VisualCompare(path_out, path_gt, path_ref, slice=44):
    """loads target and the output segmentation and
    visualizes them in the same size (ie cuts borders of target).
    TODO Slider to move through z-direction for 3Dimages. """
    out = np.load(path_out, allow_pickle=True)
    gt = np.load(path_gt, allow_pickle=True)
    ref = np.load(path_ref, allow_pickle=True)[0] #tking fat img as ref
    out = np.pad(out, pad_width=8, mode='constant', constant_values=0) #hardcoded padding of output back to original size
    #print("sizes: ", out.shape, gt.shape, ref.shape)
    print("Nr of detected organ pixels: ", np.sum(out==1), np.sum(out==2), np.sum(out==3), np.sum(out==4), np.sum(out==5))
    plt.figure(figsize=(15, 20))
    #plt.subplot(1, 3, 1)
    #plt.imshow(ref[:, slice, :].squeeze(), cmap="gray")
    #plt.subplot(1, 3, 2)
    #plt.imshow(gt[:, slice, :].squeeze(), vmin=0, vmax=5)
    plt.subplot(1, 3, 3)
    plt.imshow(out[:, slice, :].squeeze(), vmin=0, vmax=5)


