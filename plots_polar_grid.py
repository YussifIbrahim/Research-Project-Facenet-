# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:14:45 2020

@author: Ucif
"""
import keras
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import pandas as pd
import math
from my_triple_loss import load_raw_data,form_test_data,compute_dist,compute_probs


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),title="Disease Class", fontsize=10)
mark = ["o","v","^","<",">","*","s","x","D","1","2","3","4","X","+","_","d"]
col= ["gold","indigo","yellow","blue","orange","pink","chartreuse","brown","green","gray","black","purple","cyan","magenta","olive","red","aqua"]

def plot(Y_val,embeddings):

	'''
	plots the samples using polar grids
	arguments:
	Y_val--the labels of the embeddings to be plotted. this is used in forming the lengend
	embeddings--the embeddings to be plotted
	'''
    d =plt.subplot(projection='polar')

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 15
    font_size = 7
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = font_size
    plt.rcParams["legend.loc"] = "lower right"
    
    
    #mark = ["o","v","^","<",">","*","s","x","D","1","2","3","4","X","+","_","d"]
    col= ["gold","indigo","yellow","blue","orange","pink","chartreuse","brown","green","gray","black","purple","cyan","magenta","olive","red","aqua"]
    
    
    for x in range(len(Y_val)):
        #finds the x and y cordinates for all samples in the dataset supplied for plotting
        embed_1d = embeddings[x,0]
        embed_2d = embeddings[x,1]
        angle = math.atan((embed_2d/embed_1d))
        
        if (embed_1d < 0 and embed_2d> 0) or (embed_1d < 0 and embed_2d< 0):
            angle = angle + math.radians(180)
        angle = angle - math.radians(135)
        r = Y_val[x]
        if math.isnan(embed_1d):
            continue
			
		#no samples exist for class 11,12,13 so the radius for these classes are set to 14,15,16
        elif Y_val[x]==14:
            r = 11
        elif Y_val[x]==15:
            r = 12
        elif Y_val[x]==16:
            r = 13
        d.scatter(angle,r,color=col[int(Y_val[x])], label=Y_val[x],s=150,alpha=0.6)
        
      
    d.set_rticks([1,2,3,4,5,6,7,8,9,10,11,12,13])
    
    
    
    
    d.set_rmax(13)
    d.grid(True)
    legend_without_duplicate_labels(d)
    
    
    #a.legend(title="Disease Class",fontsize=10)
    plt.ylabel("first dimension",fontsize=15)
    plt.xlabel("second dimension",fontsize=15)
    plt.savefig('polar_grid.png')
    plt.show()



if __name__== "__main__":
	#load the dataset to be used
    train,val,Y_train,Y_val,Y_train_original,_= load_raw_data(train_size=451)
	#loads the the trained model to be used
    network = keras.models.load_model('best.h5')
	#calculates the embeddings
    embeddings = network.predict(train)
    plot(Y_train,embeddings)
    
