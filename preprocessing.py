#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import h5py


# In[2]:


nib.Nifti1Header.quaternion_threshold = -7e-07


# In[3]:


def save_itk(image, filename):
    im = sitk.GetImageFromArray(image, isVector=False)
    sitk.WriteImage(im, filename, True) 


# In[4]:


def crop_nii(train_imgs, train_labels,s,l):
    cropped_brain = np.zeros((l,152,152,s))
    cropped_labels = np.zeros((l,152,152,s))
    for i in range(l):
        
        for j in range(s):
            cropped_brain_slice = train_imgs[i,60:212,52:204,j]
            cropped_brain[i,:,:,j] = cropped_brain_slice
            
            cropped_label_slice = train_labels[i,60:212,52:204,j]
            cropped_labels[i,:,:,j] = cropped_label_slice
    return cropped_brain, cropped_labels


# In[5]:


def crop_nii_miccai(train_imgs, train_labels):
    cropped_brain = []
    cropped_labels = []
    for i in range(len(train_imgs)):
        s = train_imgs[i].shape[2]
        cropped_brain2 = np.zeros((152,152,s))
        cropped_labels2 = np.zeros((152,152,s))
        
        for j in range(s):
            cropped_brain_slice = train_imgs[i][43:195,44:196,j]
            cropped_brain2[:,:,j]=(cropped_brain_slice)
            
            cropped_label_slice = train_labels[i][43:195,44:196,j]
            cropped_labels2[:,:,j]=(cropped_label_slice)
        cropped_brain.append(cropped_brain2)
        cropped_labels.append(cropped_labels2)
        
    return cropped_brain, cropped_labels


# In[6]:


def add_noise(train_imgs, train_labels):
    
    train_imgs_ex = [img for img in train_imgs]
    train_labels_ex = [label for label in train_labels]
    
    for i in range(len(train_imgs)):
        brain = np.copy(train_imgs[i])
       
        for j in range(128):
            slice_b = brain[:,:,j]
            row,col= slice_b.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col))
            gauss = gauss.reshape(row,col)
            noisy = slice_b + gauss
            brain[:,:,j] = noisy
            
            
        train_imgs_ex.append(brain)
        train_labels_ex.append(train_labels[i])
        
       
    train_imgs_h5 = h5py.File("train_imgs.h5", 'w') 
    train_labels_h5 = h5py.File("train_labels.h5", 'w') 
    
    train_imgs_h5.create_dataset('train_imgs_h5', data=train_imgs_ex)
    train_labels_h5.create_dataset('train_labels_h5', data=train_labels_ex)
    
    train_imgs_h5.close()
    train_labels_h5.close()
    
    #train_imgs_ex = np.array(train_imgs_ex)
    #train_labels_ex = np.array(train_labels_ex)
   
    
    #return train_imgs_ex, train_labels_ex


# In[7]:


def flip(train_imgs,train_labels):
    train_imgs_ex = [img for img in train_imgs]
    train_labels_ex = [label for label in train_labels]
    for i in range(len(train_imgs)):
        brain = np.copy(train_imgs[i])
        label = np.copy(train_labels[i])
        for j in range(128):
            brain[:,:,j] = np.fliplr(train_imgs[:,:,j])
            label[:,:,j] = np.fliplr(train_labels[:,:,j])
            
        train_imgs_ex.append(brain)
        train_labels_ex.append(label)
        
        brain = np.copy(train_imgs[i])
        label = np.copy(train_labels[i])
        for j in range(128):
            brain[:,:,j] = np.flipud(train_imgs[:,:,j])
            label[:,:,j] = np.flipud(train_labels[:,:,j])
            
        train_imgs_ex.append(brain)
        train_labels_ex.append(label)
        
    train_imgs_ex = np.array(train_imgs_ex)
    train_labels_ex = np.array(train_labels_ex)
    
    return train_imgs_ex,train_labels_ex
        


# In[8]:


def get_data(partition):
    #filepath_brain = "data/nii_normalized/"
    filepath_labels = "data/labels/nii/"
    zeros = np.zeros((256,256,25))
    train_imgs = []
    
        
    filepath_brain = "data/nii/"
    

    
    for nii_file in next(os.walk(filepath_brain))[2]:
        nii_read = nib.load(filepath_brain+nii_file)
        nif = nii_read.get_data()
        nif = nif.swapaxes(1,2).squeeze()
        #print(nif.shape)
        nif = np.dstack((zeros,nif,zeros))        
        train_imgs.append(nif)


    
    train_imgs = np.array(train_imgs)
    train_labels = []
    for nii_file in next(os.walk(filepath_labels))[2]:
        
        
        nii_read = nib.load(filepath_labels+nii_file)
        nif = nii_read.get_data()
        nif = nif.swapaxes(1,2).squeeze()
        nif = np.dstack((zeros,nif,zeros))        
        train_labels.append(nif)
        
    
   
        
    train_labels = np.array(train_labels)
    train_imgs_m = []
    train_labels_m = []

    filepath_brain_m = "data/miccai_stripped_brain/"
    filepath_labels_m = "data/MICCAI_labels/"
    zeros = np.zeros((256,256,25))
    
    for nii_file in next(os.walk(filepath_brain_m))[2]:
        nii_read = nib.load(filepath_brain_m+nii_file)
        nif = nii_read.get_data()
        #nif = nif.swapaxes(1,2).squeeze()
        #print(nif.shape)
        nif = np.dstack((zeros,nif,zeros))        
        train_imgs_m.append(nif)



    
    for nii_file in next(os.walk(filepath_labels_m))[2]:


        nii_read = nib.load(filepath_labels_m+nii_file)
        nif = nii_read.get_data()
        #nif = nif.swapaxes(1,2).squeeze()
        #print(nif.shape)
        nif = np.dstack((zeros,nif,zeros))        
        train_labels_m.append(nif)


    filepath_brain_m = "data/miccai_normalized/"
    filepath_labels_m = "data/MICCAI_labels/"
    zeros = np.zeros((256,256,25))
    
    for nii_file in next(os.walk(filepath_brain_m))[2]:
        nii_read = nib.load(filepath_brain_m+nii_file)
        nif = nii_read.get_data()
        #nif = nif.swapaxes(1,2).squeeze()
        #print(nif.shape)
        nif = np.dstack((zeros,nif,zeros))        
        train_imgs_m.append(nif)



    
    for nii_file in next(os.walk(filepath_labels_m))[2]:


        nii_read = nib.load(filepath_labels_m+nii_file)
        nif = nii_read.get_data()
        #nif = nif.swapaxes(1,2).squeeze()
        #print(nif.shape)
        nif = np.dstack((zeros,nif,zeros))        
        train_labels_m.append(nif)




    ### crop train_imgs, train_labels
    train_imgs, train_labels = crop_nii(train_imgs, train_labels,train_imgs[0].shape[2],len(train_imgs))
    train_imgs_mc, train_labels_mc = crop_nii_miccai(train_imgs_m, train_labels_m)
    #print("data without flipping:{0}".format(len(train_imgs)))
    #### Flip and add
    #train_imgs, train_labels = flip(train_imgs, train_labels)
    #train_imgs, train_labels = add_noise(train_imgs, train_labels)
    train_slice_brain = []
    train_slice_label = []
    for i in range(len(train_imgs)):
        num_slices = train_imgs[i].shape[2]
        for j in range(25,num_slices-25):
            slice_m = train_imgs[i,:,:,j].reshape((152,152,1))
            slice_l = train_imgs[i,:,:,(j-25):j]
            slice_h = train_imgs[i,:,:,j:(j+25)]
            
            slice_tot = np.dstack((slice_l,slice_m,slice_h))
            train_slice_brain.append(slice_tot)
            train_slice_label.append(train_labels[i,:,:,j])
    
#     train_slice_brain_m = []
#     train_slice_label_m = []
    for i in range(len(train_imgs_mc)):
        #print(i)
        num_slices = train_imgs_mc[i].shape[2]
        for j in range(25,num_slices-25):
            slice_m = train_imgs_mc[i][:,:,j].reshape((152,152,1))
            slice_l = train_imgs_mc[i][:,:,(j-25):j]
            slice_h = train_imgs_mc[i][:,:,j:(j+25)]

            slice_tot = np.dstack((slice_l,slice_m,slice_h))
            train_slice_brain.append(slice_tot)
            train_slice_label.append(train_labels_mc[i][:,:,j])



    print(len(train_slice_brain))
    print(len(train_slice_label))
#     print(len(train_slice_brain_m))
#     print(len(train_slice_label_m))
    
#     train_imgs_h5 = h5py.File("/scratch/prathyuakundi/data/train_imgs.h5", 'w') 
#     train_labels_h5 = h5py.File("/scratch/prathyuakundi/data/train_labels.h5", 'w') 
#     print("saving h5")
#     train_imgs_h5.create_dataset('train_imgs_h5', data=train_slice_brain)
#     train_labels_h5.create_dataset('train_labels_h5', data=train_slice_label)
#     print("saved h5")
#     print("saving npy")
#     np.save("/scratch/prathyuakundi/data/train_imgs_h5.npy",train_slice_brain)
#     np.save("/scratch/prathyuakundi/data/train_labels_h5.npy",train_slice_label)
    
#     print("saved npy")
    
#     train_imgs_h5_m = h5py.File("/scratch/prathyuakundi/data/train_imgs_m.h5", 'w') 
#     train_labels_h5_m = h5py.File("/scratch/prathyuakundi/data/train_labels_m.h5", 'w') 
#     print("saving h5")
#     train_imgs_h5_m.create_dataset('train_imgs_h5_m', data=train_slice_brain_m)
#     train_labels_h5_m.create_dataset('train_labels_h5_m', data=train_slice_label_m)
#     print("saved h5")
#     print("saving npy")
#     np.save("/scratch/prathyuakundi/data/train_imgs_h5_m.npy",train_slice_brain_m)
#     np.save("/scratch/prathyuakundi/data/train_labels_h5_m.npy",train_slice_label_m)
    
#     print("saved npy")
    
#     train_imgs_h5.close()
#     train_labels_h5.close()
    
#     train_imgs_h5_m.close()
#     train_labels_h5_m.close()
    
    if(partition == 1):
        return train_slice_brain[0:3703], train_slice_label[0:3703] #, train_slice_brain_m, train_slice_label_m
    elif(partition==2):
        return train_slice_brain[3703:7406], train_slice_label[3703:7406]
    else:
        return train_slice_brain[7406:-1], train_slice_label[7406:-1]


# In[ ]:




