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

BATCH_SIZE = 5

#create dataset
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
#            print('[{}/{} ({:.0f}%)]\tAccumulated batch loss: {:.6f}\tBatch generalized Dice: {:.6f}'.format(batch_idx * len(notr),
#                    vsehslik, 100.*batch_idx/len(train_loader), running_loss/(batch_idx+1), dice.sum()/len(notr)))
            print('[{:.0f}%]\tAccumulated batch loss: {:.6f}\tBatch generalized Dice: {:.6f}'.format(100.*batch_idx/len(train_loader),
                                                running_loss/(batch_idx+1), dice.sum()/len(notr)))
            print('dices by organ: ', dice_organs.mean(0).data.numpy())
    if (epoch+1)%val_interval==0:
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
    training_Dice.append(epoch_dice)


#EVALUATION ON IMAGE:
#-cut it in patches
#-evaluate on patches
#-sew patches back together
#So far, Dices were calc. on patches, so they don't say much...

test_subjekti = subjekti[0:2]
test_labele = labele[0:2]
overlap = 8
test_dataset = POEMDatasetTEST(test_subjekti, patchsize=50, overlap=overlap*2)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_collate)
#TODO:  does DICE and loss etc work on this? now you have no batch channel really...

test_losses = []
test_dices = []
outpath = '/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Out/out'

net.eval()

nr = 1
patchi = []
placeringar = []
for slika, name, placing in test_loader:
    slika = torch.as_tensor(slika, device=device)  #this even needed? only to_device?

    if name>nr: #this means we're done with one whole subject.
        #we need to glue all the patches appropriately:
        segmentacija = glue_patches(patchi, placeringar) #glues, also performs nn.Softmax...
        tarca = pickle.load(open(test_labele[nr-1], 'rb'))
        tarca = torch.from_numpy(tarca[overlap:-overlap, overlap:-overlap, overlap:-overlap])
        #tole vrstico visje ne bo delalo, ce ne bo 256-16 deljivo s patchsize!

        test_loss = napaka(segmentacija, tarca.long())
        test_losses.append(test_loss.item())
        dajs = dice_coeff_per_class(segmentacija, tarca, nb_classes=6)
        test_dices.append(dajs)
        # save also the predictions, to compare visually:
        with open(outpath + name + '.pickle', 'wb') as handle:
            pickle.dump(segmentacija, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #reset for counting and new subject:
        print('Subject nr ' + name + ': \nLoss {:.4f}, \t Dices {}'.format(test_loss.item(), dajs))
        nr = name.data
        patchi = []
        placeringar = []

    segm = net(slika) #morda bo tle treba dummy dimension dodat, da bo slo tole,
    patchi.append(segm) #pol pa predn rezultat appendas, se squeeznes. da bo delalo z lepljenjem.
    placeringar.append(placing)

