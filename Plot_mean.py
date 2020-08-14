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


def remove_outliers(arr):
    elements = np.array(arr)

    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)
    
    final_list = [x for x in arr if (x > mean - 2 * sd)]
    final_list = [x for x in final_list if (x < mean + 2 * sd)]
    
    return final_list
def plot_mean(val, Y_val, model, show=True):
    embeddings = model.predict(val)
    c =plt.subplot()
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    #font_size = 7
    plt.rcParams["figure.figsize"] = fig_size
    #plt.rcParams["font.size"] = font_size
    
    col= ["gold","indigo","yellow","blue","orange","pink","chartreuse","brown","green","gray","black","purple","cyan","magenta","olive","red","aqua"]
    
    dictionary ={}
    for x in range(17):
        l=[i for i in range(len(Y_val)) if Y_val[i]==x]
        radians =[]
        for y in l:
            embed_1d=embeddings[y,1]
            embed_2d=embeddings[y,0]
            
            angles = math.atan(embed_2d/embed_1d)
            if (embed_1d < 0 and embed_2d> 0) or (embed_1d < 0 and embed_2d< 0):
                angles = angles + math.radians(180)
            radians.append(angles)
        #print(radians)
        mean_angle= np.mean(radians)
        sdv = np.std(radians)
        
        
        
        
        if math.isnan(mean_angle):
                continue   
        a=1
        b=1
        degrees=math.degrees(mean_angle)
        
        new_x = (a*b)/(math.sqrt((b**2)+(a**2)*(math.tan(mean_angle))**2))
        new_y = (a*b*(math.tan(mean_angle)))/(math.sqrt((b**2)+(a**2)*(math.tan(mean_angle))**2))
        
        if 0<=degrees<90:
            new_x = new_x
            new_y = new_y
        elif 90<degrees<270:
            new_x = -new_x
            new_y = -new_y
        
        dictionary[x] = [float(new_x), float(new_y)]
        c.scatter(new_x,new_y,color=col[x], label=x,s=100,alpha=0.6)
       
    c.grid(True)
    #c.legend()
    c.legend(title="Disease Class",fontsize=10)
    plt.ylabel("first dimension",fontsize=20)
    plt.xlabel("second dimension",fontsize=20)
    plt.savefig('mean_plots.png')
    if show==True:
        plt.show()
    elif show == False:
        return dictionary

def find_class(train,Y_train,model, val, Y_val):
    dictionary = plot_mean(train,Y_train,model,show=False)
    val_embeddings = model.predict(val)
    
    classes =[]
    dist = {}
    for x in range(len(Y_val)):
        dist.clear()
        
        for y in range(16):
            if dictionary.get(y+1) is None:
                continue
            
           # print(dictionary.get(y+1))
           
            
            
            #type_float =[float(i) for i in dictionary.get(y+1)]
            distance = compute_dist(dictionary.get(y+1), val_embeddings[x] )
            dist[y+1] = distance
            
        match = min(dist, key=dist.get)
        classes.append(match)
#    
    
    
    
    
    return classes,Y_val

def accuracy(classes,Y_val):
    true = 0
    for x in range(len(classes)):
        if classes[x]==Y_val[x]:
            true+=1
    true_percent = (true/len(classes))*100
    print("The accuray of the model is {} %".format(true_percent))
    return
            

if __name__== "__main__":
    _,val,_,Y_val,Y_train_original,_= load_raw_data(train_size=0)
    train,_,Y_train,_,Y_train_original,_= load_raw_data(train_size=451)
    model = keras.models.load_model('network.h5')
    #embeddings = network.predict(train)
    #dictionary = plot_mean(train, Y_train, model,show=True)
    classes,Y_value = find_class(train,Y_train,model, train,Y_train)
    accuracy(classes,Y_val)
#    print(dictionary )
    print(Y_value)
    
    
    
