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
import pandas as pd

BATCH_SIZE = 5
#create dataset
outpath = '/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Results/'

subjekti = glob.glob('/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Data/TRAINdata/sub*'); subjekti.sort()
labele = glob.glob('/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Data/TRAINdata/lab*'); labele.sort()
dataset = POEMDataset(subjekti[0:2], labele[0:2], 25, 9, [1,2,2,1,4,2])
train_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)




# create your optimizer, network and set parameters
net = OnePathway(4, 6, dropoutrateCon=0.2, dropoutrateFC=0.5)
net = net.float()
optimizer = optim.Adam(net.parameters(), lr=0.001)
napaka = nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean') #weight za weightedxentropy, ignore_index ce ces ker klas ignorat.
epochs = 3
log_interval = 1 #na kolko batchev reportas.
val_interval = 5 #na kolko epoch delas validation.

# train on cuda if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training loop:
train_losses = []
training_Dice = []
val_losses = []
val_Dice = []

vsehslik = len(train_loader.dataset)*BATCH_SIZE
for epoch in range(epochs):
    print('Train Epoch: {}'.format(epoch))
    net.train()
    running_loss = 0.0
    epoch_loss = 0.0
    epoch_dice = 0.0
    for batch_idx, (notr, target) in enumerate(train_loader):
        notr, target = torch.as_tensor(notr, device=device), torch.as_tensor(target, device=device)
        #notr = notr.to(device)
        #target = target.to(device)
        optimizer.zero_grad()   # zero the gradient buffers
        ven = net(notr.float())
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
                                                running_loss/(batch_idx+1), dice.sum()/len(notr)))
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

print("Training finished. Saving metrics...")
dejta = np.column_stack([np.array(train_losses), np.array(training_Dice])
df = pd.DataFrame(data=dejta,    # values
            columns=np.array(['Loss', 'Dice Bckg','Dice Liver', 'Dice 2', 'Dice 3', 'Dice 4', 'Dice 5']))
df.to_csv(outpath+'DiceAndLoss.csv')

#EVALUATION ON IMAGE:
#-cut it in patches <- do this in advance!!
#-inference on patches
#-sew patches back together
#So far, Dices were calc. on patches, so they don't say much... Calc again on sewn pics!
test_subjekti = subjekti[0:2]
test_labele = labele[0:2]
patchsize = 50
overlap = 8
test_list = cut_patches(test_subjekti, patchsize, overlap*2, channels=2, outpath=outpath, input2=False)

test_dataset = POEMDatasetTEST(test_list) #TODO fix/new dataloader za cuttane slike, z on/off switchem za multiinput
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=test_collate)
#TODO!! test loader should return (batch of patches, names), where names will be names of segmented output patches!!!

test_losses = []
test_dices = []
net.eval()
for slike, names in test_loader:
    slike = torch.as_tensor(slike, device=device)  #this even needed? only to_device?

    segm = net(slike)
   # segm = nn.Softmax(dim=1)(segm)
   # segm = torch.argmax(segm, dim=1)
    #save all processed patches temporarily; without doing softmax, since you need one-hot for dice etc later
    for patch, name in enumerate(names):
        np.save(name, np.squeeze(segm[patch,...].numpy()))


#after inference is done on entire dataset, glue temporarily saved ndarrays, evaluate Dice++, save pngs.
#we need to glue all the patches appropriately:
vsitestirani = []
for subj in test_labele:
    nr = re.findall(r'.*label([0-9]*)\.pickle', subj)
    vsitestirani.append(nr[0])
    segmentacija = torch.from_numpy(glue_patches(nr[0], outpath, patchsize, overlap)) #glues one person, saves png, returns numpy one one-hot.
    #tarca = np.load(test_labele[subj]))
    tarca = pickle.load(open(subj, 'rb'))
    tarca = torch.from_numpy(tarca[overlap:-overlap, overlap:-overlap, overlap:-overlap])

    test_loss = napaka(segmentacija, tarca.long())  #napaka needs onehot, does softmax inside.
    test_losses.append(test_loss.item())
    dajs = dice_coeff_per_class(nn.Softmax(dim=1)(segmentacija), tarca, nb_classes=6) #Dice expects softmax!
    test_dices.append(dajs.data.numpy())

    #reset for counting and new subject:
    print('Subject nr ' + subj + ': \nLoss {:.4f}, \t Dices {}'.format(test_loss.item(), dajs.data.numpy()))

print("Testing finished. Saving metrics...")
dejta = np.column_stack([np.array(test_losses), np.array(test_dices])
df = pd.DataFrame(data=dejta,    # values
                  index=vsitestirani,
                columns=np.array(['Loss', 'Dice Bckg','Dice Liver', 'Dice 2', 'Dice 3', 'Dice 4', 'Dice 5']))
df.to_csv(outpath+'DiceAndLoss_Test.csv')