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



def plot_mean(val, Y_val, model, show=True):

	'''
	plot the mean of each class 

	arguments:
	val--the validation set to be plotted
	Y_val--the labels of the validation set
	model--the trained model to me used for creating the embeddings
	
	returns:
	plot--a plot of the mean
	or
	dictionary--a dictionary of the mean of each class using class as key. 
	embeddings--embeddings is the embeddings obtained from predictions
	'''
	#create embeddings
    embeddings = model.predict(val)
    c =plt.subplot()
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    #font_size = 7
    plt.rcParams["figure.figsize"] = fig_size
    #plt.rcParams["font.size"] = font_size
    
    col= ["gold","indigo","yellow","blue","orange","pink","chartreuse","brown","green","gray","black","purple","cyan","magenta","olive","red","aqua"]
    
	#form a dictionary to store the mean of each class
    dictionary ={}
	#iterate through each class
    for x in range(17):
	
		#store all samples per class x in variable l
        l=[i for i in range(len(Y_val)) if Y_val[i]==x]
        
		#variable to store the angles
        radians =[]
        for y in l:
			#find the x and y axes of the plots
            embed_1d=embeddings[y,1]
            embed_2d=embeddings[y,0]
            
            #find the angle per sample
            angles = math.atan(embed_2d/embed_1d)
            if (embed_1d < 0 and embed_2d> 0) or (embed_1d < 0 and embed_2d< 0):
                angles = angles + math.radians(180)
            radians.append(angles)
            
        #find the mean and standard 
        mean_angle= np.mean(radians)
        sdv = np.std(radians)
        
        
        
        
        if math.isnan(mean_angle):
                continue   
        a=1
        b=1
        degrees=math.degrees(mean_angle)
        
        #calculate the x and y axes of the mean of each class
        new_x = (a*b)/(math.sqrt((b**2)+(a**2)*(math.tan(mean_angle))**2))
        new_y = (a*b*(math.tan(mean_angle)))/(math.sqrt((b**2)+(a**2)*(math.tan(mean_angle))**2))
        
        #make transformations to the x and y values based on the angle
        if 0<=degrees<90:
            new_x = new_x
            new_y = new_y
        elif 90<degrees<270:
            new_x = -new_x
            new_y = -new_y
            
        #add the mean to the dictionary using the class as key
        dictionary[x] = [float(new_x), float(new_y)]
        
		#plot the mean 
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
        return dictionary,embeddings

def find_class(train,Y_train,model, val, Y_val):

	'''
	finds the accuracy of the trained model. this is done by comparing the distance between the validation samples and the mean of the 
	trained set	calculated by plot_mean function above. The new class belongs to the class with the smallest distance between them
	 arguments:
	 train--the train set used for training the model
	 Y_train--the labels of the train set
	 model--the trained model to be used for prediction
	 val--the dataset used to test the accuracy
	 Y_val--the class labels of the test set
	 
	 return:
	 classes--the class with the smallest distance to the sample
	 Y_val--the true class of the sample
	 
	 
	 
	'''
    dictionary ,val_embeddings= plot_mean(train,Y_train,model,show=False)
    #a list to store the class with smallest distance 
    classes =[]
    
	# a dictionary to store the distance for each sample
    dist = {}
    for x in range(len(Y_val)):
        dist.clear()
        
        for y in range(16):
            if dictionary.get(y+1) is None:
                continue
            
        
            #type_float =[float(i) for i in dictionary.get(y+1)]
            distance = compute_dist(dictionary.get(y+1), val_embeddings[x] )
            dist[y+1] = distance
            
        match = min(dist, key=dist.get)
        classes.append(match)

    return classes,Y_val

def accuracy(classes,Y_val):

	'''
	calculates the accuracy of results from  find_class() function above
	'''
    true = 0
    for x in range(len(classes)):
        if classes[x]==Y_val[x]:
            true+=1
    true_percent = (true/len(classes))*100
    print("The accuray of the model is {} %".format(true_percent))
    return
            

if __name__== "__main__":
	#loads the train and test data with corresponding labels
    _,val,_,Y_val,Y_train_original,_= load_raw_data(train_size=0)
	#loads trained model
     model = keras.models.load_model('best.h5')
    dictionary,embeddings = plot_mean(train, Y_train, model,show=True)
    classes,Y_value = find_class(train,Y_train,model, train,Y_train)
    accuracy(classes,Y_val)

    
    
    
