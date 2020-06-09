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

#set parameters
BATCH_SIZE = 6
epochs = 2
sampling = [1,5,3,2,5,4] #0 bckg, 1 bladder, 2 kidneys, 3 liver, 4 pancreas, 5 spleen
outpath = '/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Results/'

subbatch = sum(sampling)

continue_training = True #set to true if you want to load a net and train it further from there on. If yes, give string
what_to_load = "2020-04-14 22:00:37.557746" #string of the time as the unique id of the net you want to load

#set dataset and loaders
subjekti = glob.glob('/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Data/TRAINdata/sub*'); subjekti.sort()
labele = glob.glob('/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Data/TRAINdata/lab*'); labele.sort()
#dataset = POEMDataset(subjekti[0:10], labele[0:10], 25, 9, sampling)
dataset = POEMDatasetMultiInput(subjekti[0:10], labele[0:10], sampling, 25, channels=[0,1], subsample=True,
                 channels_sub=[0,1])
train_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=multi_input_collate) #my_collate)



# create your optimizer, network and set parameters

#net = OnePathway(in_channels=4, num_classes=6, dropoutrateCon=0.2, dropoutrateFC=0.5)
net = DualPathway(in_channels_orig=2, in_channels_subs=2, num_classes=6, dropoutrateCon=0.2, dropoutrateFC=0.5, nonlin=nn.PReLU())
#net= DualPathwayNico(in_channels_orig=2, in_channels_subs=2, num_classes=6, dropoutrateCon=0.2, dropoutrateFC=0.5, nonlin=nn.PReLU())
net = net.float()

optimizer = optim.Adam(net.parameters(), lr=0.001)

#napaka = nn.CrossEntropyLoss(weight=None, ignore_index=0, reduction='mean') #weight za weightedxentropy, ignore_index ce ces ker klas ignorat.
napaka = SoftDiceLoss(nb_classes=6, weight=np.array([1., 1., 1., 1., 2., 1.]))

prev_epochs = 0
sampling_history = {}
past_checkpoints = []
if continue_training:
    checkpoint = torch.load(outpath + "Networks/" + net.name + "_" + what_to_load +".pt")
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    sampling_history = checkpoint['sampling_history']
    prev_epochs = checkpoint['all_epochs']
    past_checkpoints = checkpoint['past_checkpoints']



log_interval = 1 #na kolko batchev reportas.
val_interval = 5 #na kolko epoch delas validation.
# train on cuda if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training loop:
train_losses = []
training_Dice = []
val_losses = []
val_Dice = []

vsehslik = len(train_loader.dataset)*subbatch

for epoch in range(epochs):
    print('Train Epoch: {}'.format(epoch))
    net.train()
    running_loss = 0.0
    epoch_loss = 0.0
    epoch_dice = 0.0

  #  tqdm_iter = tqdm_(enumerate(train_loader), total=len(train_loader), desc=">>Training: ")
    for batch_idx, (notr, target) in enumerate(train_loader): #tqdm_iter:
        notr, target = [torch.as_tensor(notri, device=device).float() for notri in notr], torch.as_tensor(target, device=device)
        #notr = notr.to(device)
        #target = target.to(device)
       # print(len(notr))
        optimizer.zero_grad()   # zero the gradient buffers
        ven = net(*notr)
        loss = napaka(ven, target.long())
        loss.backward()
        optimizer.step()

        dice = dice_coeff(nn.Softmax(dim=1)(ven), target, nb_classes=6, weights=np.array([0,1,1,1,1,1]))
        epoch_loss += loss.item() * ven.shape[0] #or loss.detach() - detach disables differentiation through here (?)
        dice_organs = dice_coeff_per_class(nn.Softmax(dim=1)(ven), target, nb_classes=6)
        epoch_dice += dice_organs.sum(0).squeeze()
        running_loss += loss.item()
        if batch_idx%log_interval==0:
            print('[{:.0f}%]\tAccumulated batch loss: {:.6f}\tBatch generalized Dice: {:.6f}'.format(100.*batch_idx/len(train_loader),
                                                running_loss/(batch_idx+1), dice.sum()/len(notr[0])))
            print('dices batch averages by organ: ', dice_organs.mean(0).data.numpy())

    if (epoch+1)%val_interval==0: #TODO add validation
        #with torch.no_grad():
            #for x_val, y_val in val_loader:
             #   x_val = x_val.to(device)
            #    y_val = y_val.to(device)
           #     net.eval()
          #      yhat = model(x_val)
         #       val_loss = loss_fn(y_val, yhat)
        #        val_losses.append(val_loss.item())
        pass
                #do validation, save results. PRint.

    epoch_loss = epoch_loss/vsehslik
    epoch_dice = epoch_dice.squeeze()/vsehslik
    print('Epoch {} finished. Averaged loss: {:.6f}, average Dices: {} \n'.format(epoch, epoch_loss, epoch_dice.data.numpy()))
    train_losses.append(epoch_loss)
    training_Dice.append(epoch_dice.data.numpy())

#add also infos on sampling of the given run to history:
sampling_history[prev_epochs+epochs] = sampling

print("Training finished. Saving metrics...")
cas = dt.datetime.now()
dejta = np.column_stack([np.array(train_losses), np.array(training_Dice)])
df = pd.DataFrame(data=dejta,    # values
    columns=np.array(['Loss', 'Dice Bckg', 'Dice Bladder', 'Dice Kidneys', 'Dice Liver', 'Dice Pancreas', 'Dice Spleen']))
df.to_csv(outpath + f'DiceAndLoss_{cas}.csv')
print("Saving trained network...")
#TODO: saving (and reloading) of network
checkpoint = {
    'past_checkpoints': past_checkpoints + [str(cas)],
    'all_epochs': epochs+prev_epochs,
 #   'valid_loss_min': valid_loss,
    'state_dict': net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'sampling_history': sampling_history
}
torch.save(checkpoint, outpath + "Networks/" + net.name + "_" + str(cas)+".pt")




#TODO if needed: now you load a network and train a bit more, the total metrics will be split into different CSVs. So ...
# write a function that loads torch history to get a list of all checkpoints, and then plots the joint training history?

