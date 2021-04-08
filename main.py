#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:17:05 2019

@author: eva
"""

import glob
from generators import *
from helpers import *
from networks import *
import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
np.set_printoptions(precision=3, suppress=True)
#import hiddenlayer as hl

######################################################################################################
#               SET PARAMETERS
######################################################################################################
BATCH_SIZE = 4 #8
epochs = 10
#sampling = [1,5,2,2,5,4,2] # [bckg, bladder, R kidney, liver, pancreas, spleen, L kidney]
sampling = [0,4,2,2,4,3,2]
outpath = 'Results/' #'/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Results/'

orig_channels = [0,1] #which channels to use in orig pathway

use_subsamp = True #do we use subsampled pathway?
subsamp_channels = [0,1] # None #[0,1] #which channels to use in subsampled pathway.
assert use_subsamp!=(subsamp_channels is None)

add_chan_size = None #11 #do we have yet another pathway? If yes, how big is the segmentsize? Else NONE!!!
add_channels = [2,3] #which channels to use in this additional pathway, if used.
add_chan_FM_sizes = [30] #List of feature map sizes in the added pathway, if used. 
add_join_at = 5 #At which layer in the pathway to fuse the added pathway, if in use. 
add_to_orig = True #True if fusing into the original pathway, False if fusing to subsampled.

#some checks:
assert ((add_chan_size is None) or ((1 + 2*len(add_chan_FM_sizes))<=add_chan_size)), "Segment size in 3rd pathway too small for the given FM list!"
assert 0 <= add_join_at < 10
nr_orig = len(orig_channels)
nr_subs = len(subsamp_channels) if use_subsamp else None
nr_add = None if add_chan_size is None else len(add_channels)
#
continue_training = False #set to true if you want to load a net and train it further from there on. If yes, give string what_to_load
what_to_load = "2020-04-14 22:00:37.557746" #string of the time as the unique id of the net you want to load
#

dataPath = 'Data/' #'/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Data/'
num_classes = 7
######################################################################################################

subbatch = sum(sampling)
# get dataset and loaders
subjekti = glob.glob(dataPath + 'TRAINdata/sub*'); subjekti.sort()
labele = glob.glob(dataPath + 'TRAINdata/lab*'); labele.sort()
subjekti_val = glob.glob(dataPath + 'VALdata/sub*'); subjekti_val.sort()
labele_val = glob.glob(dataPath + 'VALdata/lab*'); labele_val.sort()

#####################################################################################################
#               NETWORK AND TRAINING OPTIONS
#####################################################################################################
dataset = POEMDatasetMultiInput(subjekti, labele, sampling, segment_size=25, channels=orig_channels, subsample=use_subsamp,
                 channels_sub=subsamp_channels, segment_size2=add_chan_size, channels2=add_channels, num_classes=num_classes)
dataset_val = POEMDatasetMultiInput(subjekti_val, labele_val, sampling=[1,1,1,1,1,1,1], segment_size=25, channels=orig_channels, subsample=use_subsamp,
                 channels_sub=subsamp_channels, segment_size2=add_chan_size, channels2=add_channels, num_classes=num_classes)
                 #POEMDatasetTEST(subjekti_val, labele_val, channels=use_channels, subsampled=use_subsamp, 
                #channels_sub=subsamp_channels, input2=add_chan_size, channels2=add_chans)
train_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=multi_input_collate, num_workers=BATCH_SIZE+5, pin_memory=True)
val_loader = data.DataLoader(dataset_val, batch_size=BATCH_SIZE, collate_fn=multi_input_collate, num_workers=BATCH_SIZE+5, pin_memory=True)


# create your optimizer, network and set parameters
#net = OnePathway(in_channels=nr_orig, num_classes=num_classes, dropoutrateCon=0.2, dropoutrateFC=0.5)
net = DualPathway(in_channels_orig=nr_orig, in_channels_subs=nr_subs, num_classes=num_classes, 
                    dropoutrateCon=0.2, dropoutrateFC=0.5, nonlin=nn.PReLU())
#net = MultiPathway(in_channels_orig=nr_orig, in_channels_subs=nr_subs, in_channels_add=nr_add, 
#                    join_at=add_join_at, join_to_orig=add_to_orig, add_FM_sizes=add_chan_FM_sizes, num_classes=7, dropoutrateCon=0.2, dropoutrateFC=0.5, nonlin=nn.PReLU())

net = net.float()
net.apply(weights_init)

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99), amsgrad=False)

#napaka = nn.CrossEntropyLoss(weight=None, ignore_index=0, reduction='mean') #weight za weightedxentropy, ignore_index ce ces ker klas ignorat.
#napaka = SoftDiceLoss(nb_classes=num_classes, weight=np.array([1, 1, 1, 1, 2, 1, 1]))

#napaka1 = nn.CrossEntropyLoss(weight=None, ignore_index=-1, reduction='mean')
#napaka2 = SoftDiceLoss(nb_classes=num_classes, weight=np.array([0, 1, 1, 1, 1, 1, 1]))

#napaka1 = DiceLoss(nb_classes=num_classes, weight=np.array([0, 2, 1, 2, 2, 1, 1]))
#napaka2 = CrossEntropy(nb_classes=num_classes, weight=np.array([1, 1, 1, 2, 1, 1, 1]))
#######################################################################################################

np.set_printoptions(precision=3, suppress=True) #for prettier printing

prev_epochs = 0
sampling_history = {}
past_checkpoints = []
if continue_training:
    checkpoint = torch.load(outpath + "Networks/" + net.name + "_" + what_to_load +".pt")
    assert checkpoint['name'] == net.name
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    sampling_history = checkpoint['sampling_history']
    prev_epochs = checkpoint['all_epochs']
    past_checkpoints = checkpoint['past_checkpoints']



#log_interval = 100 #na kolko batchev reportas.
val_interval = 1 #na kolko epoch delas validation.
# train on cuda if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}...")

napaka = SoftDiceLoss(nb_classes=num_classes, weight=torch.tensor([0, 2, 2, 3, 1, 2, 2], device=device))
napaka1 = DiceLoss(nb_classes=num_classes, weight=torch.tensor([0, 2, 1, 2, 2, 1, 1], device=device))
napaka2 = CrossEntropy(nb_classes=num_classes, weight=torch.tensor([1, 1, 1, 2, 1, 1, 1], device=device))


# training loop:
training_losses = []
training_Dice = []
val_losses = []
val_Dice = []
val_epoche = []

vsehslik = len(train_loader.dataset)*subbatch
net.to(device)
dl, cen = 0.8, 0.8
for epoch in range(epochs):
    print('\nEpoch: {}'.format(epoch))
    net.train()
    running_loss = 0.0
    epoch_loss = 0.0
    epoch_dice = 0.0


    tqdm_iter = tqdm(train_loader, total=len(train_loader), desc=">>Training: ", leave=False)
  #  for batch_idx, (notr, target) in enumerate(train_loader): #tqdm_iter:
    for (notr, target) in tqdm_iter:
        notr, target = [torch.as_tensor(notri, device=device).float() for notri in notr], torch.as_tensor(target, device=device)
        #notr = notr.to(device)
        #target = target.to(device)
       # print(len(notr))
        #print(target.shape)
        optimizer.zero_grad()   # zero the gradient buffers
        ven = net(*notr)
       # print((ven.shape, target.shape))

    #    loss = napaka(ven, target.long())
        loss = dl*napaka1(ven, target.long()) + cen*napaka2(ven, target.long())
        loss.backward()
        optimizer.step()

        dice = dice_coeff(nn.Softmax(dim=1)(ven.detach()), target.detach(), nb_classes=num_classes, weights=torch.tensor([0.,1.,1.,1.,1.,1.,1.], device=device))
        epoch_loss += loss.item() * ven.shape[0] #or loss.detach() - detach disables differentiation through here (?)
        dice_organs = dice_coeff_per_class(nn.Softmax(dim=1)(ven.detach()), target.detach(), nb_classes=num_classes)
        epoch_dice += dice_organs.sum(0).squeeze()
        running_loss += loss.item()
  #      if batch_idx%log_interval==0:
  #          print('[{:.0f}%]\tAccumulated batch loss: {:.6f}\tBatch generalized Dice: {:.6f}'.format(100.*batch_idx/len(train_loader),
  #                                              running_loss/(batch_idx+1), dice.sum()/len(notr[0])))
  #          print('dices batch averages by organ: ', dice_organs.mean(0).data.numpy())


    epoch_loss = epoch_loss/vsehslik
    epoch_dice = epoch_dice/vsehslik
    
    print(f">> TRAINING: \n Averaged loss:  {epoch_loss:.4f}, avg Dices: {epoch_dice.detach().cpu().numpy()}")
    training_losses.append(epoch_loss)
    training_Dice.append(epoch_dice.detach().cpu().numpy())


    if (epoch+1)%val_interval==0: 
        val_epoche.append(epoch)
        val_loss_rolling = 0.0
        val_Dice_rolling = 0.0
        val_batches = len(val_loader)
        tq_iter_val = tqdm(val_loader, total=val_batches, desc=">>Validation: ", leave=False)
        with torch.no_grad():
            for x_val, y_val in tq_iter_val: #val_loader:
                x_val = [torch.as_tensor(xv,device=device).float() for xv in x_val]
                y_val = torch.as_tensor(y_val, device=device)
                net.eval()
                y_hat = net(*x_val)

                val_loss_rolling += napaka1(y_hat, y_val.long()).item() + napaka2(y_hat, y_val.long()).item()
                val_Dice_rolling += dice_coeff_per_class(nn.Softmax(dim=1)(y_hat), y_val, nb_classes=num_classes).mean(0).squeeze()

            val_losses.append(val_loss_rolling/val_batches)

            val_Dice.append(val_Dice_rolling.detach().cpu().numpy()/val_batches)
        print(f'>> VALIDATION: \n Total val loss: {val_losses[-1]:.4f},  val Dices: {val_Dice[-1]}')
    
    dl, cen = min(dl+0.01, 1), max(cen-0.01, 0.01) #update weighted loss



#unique identifier:
cas = dt.datetime.now()
   
#add also infos on sampling of the given run to history:
sampling_history[prev_epochs+epochs] = sampling

print("Training Done. Saving metrics...")
dejta_tr = np.column_stack([np.array(training_losses), np.array(training_Dice)])
df = pd.DataFrame(data=dejta_tr,    # values
    columns=np.array(['Loss', 'Dice Bckg', 'Dice Bladder', 'Dice R Kidney', 'Dice Liver', 'Dice Pancreas', 'Dice Spleen', 'Dice L Kidney']))
df.to_csv(outpath + f'DiceAndLoss_{cas}_train.csv')
dejta_vl = np.column_stack([np.array(val_epoche), np.array(val_losses), np.array(val_Dice)])
df = pd.DataFrame(data=dejta_vl,    # values
    columns=np.array(['Epoch', 'Loss', 'Dice Bckg', 'Dice Bladder', 'Dice R Kidney', 'Dice Liver', 'Dice Pancreas', 'Dice Spleen', 'Dice L Kidney']))
df.to_csv(outpath + f'DiceAndLoss_{cas}_val.csv')


#TODO: save loss type and weights somewhere too. 
print("Saving trained network...")
checkpoint = {
    'past_checkpoints': past_checkpoints + [str(cas)],
    'all_epochs': epochs+prev_epochs,
 #   'valid_loss_min': valid_loss,
    'state_dict': net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'sampling_history': sampling_history,
    'name': net.name
}
torch.save(checkpoint, outpath + "Networks/" + net.name + "_" + str(cas)+".pt")
print(f"Done. Saved under id: {cas} ({net.name})")


