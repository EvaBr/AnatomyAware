#EVALUATION ON IMAGE:
#first set parameters. Most important is CAS, as in unique identifier, to load the net to be evaluated.
# TODO separate file. function should take in CAS, SUBJECTS, LABELE, NET....
#-cut it in patches <- do this in advance!!
#-inference on patches
#-sew patches back together
#So far, Dices were calc. on patches, so they don't say much... Calc again on sewn pics!
test_subjekti = subjekti[0:5]
test_labele = labele[0:5]
patchsize = 52
overlap = 8
need_to_cut = False #True #change to True if any of test data parameters are changed.
do_inference = True
outpath = PATCHES #TODO

if need_to_cut:
    test_list = cut_patches(test_subjekti, None, patchsize, overlap*2, channels=4, outpath=outpath, subsampledinput=True) #test_labele = None
    # best to always run 'cut_patches' with 4channels, since data loader itself takes care of cases with using different channels.
else:
    test_list = glob.glob(outpath+"subj_*[0-9].npy")

test_dataset = POEMDatasetTEST(test_list, channels=[0,1], subsampled=True, channels_sub=[0,1], input2=None, channels2=None)
test_loader = data.DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_collate)

test_losses = []
test_dices = []
#TODO LOAD NET
net.eval()
net.to(device)
i=0
if False: #do_inference:
#with torch.no_grad(): #TODO
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
vsitestirani = []
for subj in test_labele:
    nr = re.findall(r'.*label([0-9]*)\.pickle', subj) #TODO! NOW WE HAVE NUMPY
    print("Saving tests on subject nr ", nr)
    vsitestirani.append(nr[0])
    segmentacija = torch.from_numpy(glue_patches(nr[0], outpath, patchsize, overlap, nb_classes=7)) #glues one person, saves png, returns numpy one one-hot.
    #tarca = np.load(test_labele[subj]))
    tarca = pickle.load(open(subj, 'rb')) #TODO NUMPY
    tarca = torch.from_numpy(tarca[np.newaxis, overlap:-overlap, overlap:-overlap, overlap:-overlap])

    test_loss = napaka(segmentacija, tarca.long())  #napaka needs onehot, does softmax inside.
    test_losses.append(test_loss.item())
    dajs = dice_coeff_per_class(nn.Softmax(dim=1)(segmentacija), tarca, nb_classes=7) #Dice expects softmax!
    test_dices.append(dajs.data.numpy().squeeze())

    #Now also save a vtk of not one-hot, so you can compare in 3dslicer!
    #TODO

    #reset for counting and new subject:
    print('Loss {:.4f}, \t Dices {}'.format(test_loss.item(), dajs.data.numpy()))

print("Testing finished. Saving metrics...")
dejta = np.column_stack([np.array(test_losses), np.array(test_dices)])
df = pd.DataFrame(data=dejta,    # values
                  index=vsitestirani,
                  columns=np.array(['Loss', 'Dice Bckg', 'Dice Bladder', 'Dice Kidneys', 'Dice Liver', 'Dice Pancreas', 'Dice Spleen']))
df.to_csv(outpath+f'DiceAndLoss_Test_{cas}.csv')






