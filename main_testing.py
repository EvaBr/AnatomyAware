#EVALUATION ON IMAGE:
#first set parameters. Most important is CAS, as in unique identifier, to load the net to be evaluated.
# TODO separate file. function should take in CAS, SUBJECTS, LABELE, NET....
#-cut it in patches <- do this in advance!!
#-inference on patches
#-sew patches back together
#So far, Dices were calc. on patches, so they don't say much... Calc again on sewn pics!
import glob
import numpy as np 
import torch 
import re
from networks import *
from helpers import *
from generators import *

################ WHICH NET+DATA TO TEST
dataPath = '/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Data/TESTdata/'
networkPath = '/home/eva/Desktop/research/PROJEKT2-DeepLearning/AnatomyAwareDL/Results/Networks/'
what_to_load = "netname.pt" #string of the time as the unique id of the net you want to load
num_classes = 7

patchsize = 52
overlap = 8

need_to_cut = False #True #change to True if any of test data parameters are changed.
do_inference = True #turn to false if you just want to cut images
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

outpath = dataPath+'PATCHES/'
test_subjekti = glob.glob(dataPath + "subj_*.npz"); test_subjekti.sort()
test_labele = glob.glob(dataPath + "label*.npy"); test_labele.sort()

if need_to_cut:
    test_list = cut_patches(test_subjekti, patchsize, overlap*2, channels=4, outpath=outpath, subsampledinput=True) #test_labele = None
    # best to always run 'cut_patches' with 4channels, since data loader itself takes care of cases with using different channels.
else:
    test_list = glob.glob(outpath+"subj_*[0-9].npy")

use_channels = [0,1]
subsamp_channels = [2,3]
additional_channels = None
test_dataset = POEMDatasetTEST(test_list, channels=use_channels, subsampled=True, channels_sub=subsamp_channels, input2=None, channels2=None)
test_loader = data.DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_collate)


#LOAD NET:
checkpoint = torch.load(networkPath + what_to_load)
net = DualPathway(in_channels_orig=len(use_channels), in_channels_subs=len(subsamp_channels), num_classes=num_classes, 
                    dropoutrateCon=0.2, dropoutrateFC=0.5, nonlin=nn.PReLU())
net = net.float()
net.load_state_dict(checkpoint['state_dict'])
net.to(device)
net.eval()
napaka = SoftDiceLoss(nb_classes=num_classes, weight=np.array([1., 1., 1., 1., 2., 1., 1.])) #which loss, together with per-organ dice 
# you want to evaluate on the test set. Advised to be the same as when training.

i=0
#if do_inference:
with torch.no_grad(): 
    for slike, names in test_loader:
        print(f"Doing test inference, batch nr. {i+1}/{len(test_loader)}")
        i+=1
        slike = [torch.as_tensor(slika, device=device) for slika in slike]  #this even needed? only to_device?
  #     print("len slike: ", len(slike))
  #     print("shape slike[0]: ", slike[0].shape)
        segm = net(*slike)
        #save all processed patches temporarily; without doing softmax, since you need one-hot for dice etc later
        for patchnr, name in enumerate(names):
            np.save(name, np.squeeze(segm[patchnr,...].detach().numpy()))
            #ce ne dam detach se kurac prtozi: RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.


#after inference is done on entire dataset, glue temporarily saved ndarrays, evaluate Dice++, save pngs.
#we need to glue all the patches appropriately:
test_losses = []
test_dices = []
vsitestirani = []
for subj in test_labele:
    nr = re.findall(r'.*label_([0-9]*)\.npy', subj) #TODO! NOW WE HAVE NUMPY
    print("Saving tests on subject nr ", nr)
    vsitestirani.append(nr[0])
    segmentacija = torch.from_numpy(glue_patches(nr[0], outpath, patchsize, overlap, nb_classes=num_classes)) #glues one person, saves png, returns numpy one one-hot.
    tarca = np.load(subj)
    tarca = torch.from_numpy(tarca[np.newaxis, overlap:-overlap, overlap:-overlap, overlap:-overlap])

    test_loss = napaka(segmentacija, tarca.long())  #napaka needs onehot, does softmax inside.
    test_losses.append(test_loss.item())
    dajs = dice_coeff_per_class(nn.Softmax(dim=1)(segmentacija), tarca, nb_classes=num_classes) #Dice expects softmax!
    test_dices.append(dajs.data.numpy().squeeze())

    #Now also save a vtk/png/sth of not one-hot, so you can compare in 3dslicer! <- well, seems nontrivial. just save as npy instead for now.
    #actually, it is already saved  inside the glue_patches!

    #reset for counting and new subject:
    print('Loss {:.4f}, \t Dices {}'.format(test_loss.item(), dajs.data.numpy()))

print("Testing finished. Saving metrics... OBS! Always saved under the same name!")
dejta = np.column_stack([np.array(test_losses), np.array(test_dices)])
df = pd.DataFrame(data=dejta,    # values
                  index=vsitestirani,
                  columns=np.array(['Loss', 'Dice Bckg', 'Dice Bladder', 'Dice Kidney Left', 'Dice Liver', 'Dice Pancreas', 'Dice Spleen', 'Dice Kidney Right']))
df.to_csv(outpath+f'DiceAndLoss_Test.csv')



#to visualize an example subject, do:
nr = 22 #nr of subject to load
GT = np.load(dataPath+f'label_{nr}.npy')
out = np.load(f'{outpath}results/out{nr}.npy')
ref = np.load(dataPath+f'subj_{nr}.npz')['channels'][1, ...] #take just the first channel as the fat reference scan
compareimages(GT, out, ref)
#or 
VisualCompare(GT, out, ref, slice=44)


