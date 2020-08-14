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
    ax.legend(*zip(*unique),title="Disease Class",fontsize=10)
mark = ["o","v","^","<",">","*","s","x","D","1","2","3","4","X","+","_","d"]
col= ["gold","indigo","yellow","blue","orange","pink","chartreuse","brown","green","gray","black","purple","cyan","magenta","olive","red","aqua"]

def plot(Y_val,embeddings):
   

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    font_size = 7
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = font_size
    
    
    #mark = ["o","v","^","<",">","*","s","x","D","1","2","3","4","X","+","_","d"]
    col= ["gold","indigo","yellow","blue","orange","pink","chartreuse","brown","green","gray","black","purple","cyan","magenta","olive","red","aqua"]
    
    c =plt.subplot()
    for x in range(len(Y_train)):
        
       
        embed_1d=embeddings[x,1]
        embed_2d=embeddings[x,0]
            
           
        
        c.scatter(embed_2d,embed_1d,color=col[int(Y_train[x])], label=Y_val[x],s=100,alpha=0.6)
        

    #c.legend()
    #c.legend(title="Disease Class",fontsize=20)
    
     
    
    
    
    c.grid(True)
    legend_without_duplicate_labels(c)
    
    
    #a.legend(title="Disease Class",fontsize=10)
    plt.ylabel("first dimension",fontsize=15)
    plt.xlabel("second dimension",fontsize=15)
    plt.savefig('allplots-epoch300.pdf')
    plt.show()



if __name__== "__main__":
    train,val,Y_train,Y_val,Y_train_original,_= load_raw_data(train_size=451)
    network = keras.models.load_model('best.h5')
    embeddings = network.predict(train)
    plot(Y_train,embeddings)
    #print(embeddings)



   


