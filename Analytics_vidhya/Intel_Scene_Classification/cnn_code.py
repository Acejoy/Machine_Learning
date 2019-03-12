#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 03:32:11 2019

@author: srinjoy
"""

import os
import math
import time
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from torch import optim
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader


CUDA=False
epochs=30
seed=1
hidden_size=20
intermediate_size=128
batch_size=2
log_interval=10


torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
    
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}    

#dictionary for labels_indices to labels
indices_to_labels={0:'buildings', 1: 'forest' ,2:'glacier' ,3:'mountain' ,4: 'sea', 5:'street'}


#interactive mode on
plt.ion()

#%%

#Defining the transforms
data_transforms=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])

image_data_dir='/home/srinjoy/Documents/machine_learning/self/Analytics_VIdhya/Intel_Scene_Classification/train-scene classification/train'

data_dir='/home/srinjoy/Documents/machine_learning/self/Analytics_VIdhya/Intel_Scene_Classification/'

#change the current directory too data_dir
os.chdir(data_dir)


#read the names and indices of train and test images from csv files
data_frames={'train' : pd.read_csv('train.csv'), 'test' : pd.read_csv('test_WyRytb0.csv') }


#functio that takes the image path ,reads it and applies transforms
def get_image(path,trans):
        image_path=os.path.join(image_data_dir,path)
        image=Image.open(image_path)
        image=trans(image)
        return image
        
#plot the points and annotate them
def plot_graph(A,B,plt_str,bool_grid):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        plt.plot(A,B,plt_str)
        for x,y in zip(A,B):
                ax.annotate('(%s,%s)' %(x,y),xy=(x,y),textcoords='data')
        if bool_grid:
                plt.grid()
        plt.show()
        
#plot the points as a bar graoh and annotate the bar
def plot_bar(A,B):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.bar(A,B)
    for x,y in zip(A,B):
            ax.annotate('(%s,%s)' %(x,y),xy=(x,y),textcoords='data')
    plt.show()    
        
#%%
    
#class that creates the image dataset
class Imageset(Dataset):
        
        def __init__(self,i_dir,transforms,csv_dataframe):
                self.i_dir=i_dir
                self.transforms=transforms
                self.csv_dataframe=csv_dataframe
                self.i_dset=self.csv_dataframe.to_dict('records')
                
                #now the above dictionary contains the index a s key and (image name and label) as value
                #include the image tensor in value for each index
                
                for idx in range(len(self.i_dset)):
                        self.i_dset[idx]['image'] = get_image(self.i_dset[idx]['image_name'],self.transforms)
                        
                 
        #return the length of dataset        
        def __len__(self):
                return len(self.i_dset)
                        
        def __getitem__(self,idx):
                sample=self.i_dset[idx]
                
                return sample
        
        
#get the image dataset for train and test        
image_dataset={x:Imageset(image_data_dir,data_transforms,data_frames[x]) 
                                for x in ['train','test']}


#%%
#plot the labels vs the number of each labels in the train set

#partition the train set data_frame lable-wise and then count frequency of each label
labels=data_frames['train'].groupby('label').count()

#get the labels in the train_set
labels_index=list(labels.index)

plot_bar(labels_index,labels['image_name'])


#%%

#displaying some of the images along with their names

def display_images(image_tensor,indices_of_images):
        
        to_pil=transforms.ToPILImage()
        pil_images=[to_pil(x) for x in image_tensor]
        print(pil_images[0].size)
        num_images=len(image_tensor)
        columns=2
        rows=math.ceil(num_images/2)
        fig=plt.figure(figsize=(rows,columns))
        fig.suptitle('Images')
        for idx,idx_of_image in enumerate(indices_of_images):
                
                ax=fig.add_subplot(rows,columns,idx+1) 
                image_label=indices_to_labels[data_frames['train'].loc[idx_of_image,'label']]
                ax.set_title('(%s , %s)'% (data_frames['train'].loc[idx_of_image,'image_name'], image_label))
                plt.imshow(pil_images[idx])
        plt.show()
        
        
        
#randomly choose n number of images
size_dataset=len(image_dataset['train'].csv_dataframe)
n=random.sample(range(0,size_dataset),5)

display_image_set=[]

for x in n:
        display_image_set.append(image_dataset['train'][x]['image'])

display_images(display_image_set,n)



#%%






                



