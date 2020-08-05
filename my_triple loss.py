import time
import tensorflow as tf
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import keras
#matplotlib inline
from pylab import *
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model,normalize
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate




def load_raw_data(path = "arrhythmia_data.txt",train_size=1):
    #The arrythmia dataset can be downloaded from https://archive.ics.uci.edu/ml/datasets/Arrhythmia
    
    f = open( path, "r")
    data = []
    
    #remove line breaker, comma separate and store in array
    for line in f:
        line = line.replace('\n','').replace('?','0')
        line = line.split(",")
        
        data.append(line)
    f.close()
    
    data = np.array(data).astype(np.float64)
    print(data.shape)
    

    #create the class labels for input data
    Y_train_original = data[:,-1:]
    train = data[:,:-1]
    #split the data into train and validation set with corresponding labels
    
    val = train[train_size:,:]
    train = train[:train_size,:]
    train = train.reshape(train.shape[0],train.shape[1])
    Y_val =(Y_train_original[train_size:,:]).astype(int)
    Y_train =(Y_train_original[:train_size,:]).astype(int)
    return train,val,Y_train,Y_val,Y_train_original,train_size
    
def load_data():
    
    train,val,_,_,Y_train_original,train_size = load_raw_data()
    #create one hot encoding of the class labels of the data and separate them into train and test data
    lb = LabelBinarizer()
    encode = lb.fit_transform(Y_train_original)
    nb_classes = int(len(encode[0]))
    
    #one_hot_labels = keras.utils.to_categorical(labels, num_classes=10) this could also be used for one hot encoding
    Y_val_e = encode[train_size:,:]
    Y_train_e = encode[:train_size,:]
    
    
    
    
    val_in = []
    train_in = []
    
    #grouping and sorting the input data based on label id or name
    for n in range(nb_classes):
        images_class_n = np.asarray([row for idx,row in enumerate(train) if np.argmax(Y_train_e[idx])==n])
        train_in.append(images_class_n)
        
        
        images_class_n = np.asarray([row for idx,row in enumerate(val) if np.argmax(Y_val_e[idx])==n])
        val_in.append(images_class_n)
    
    
   #returns a lis of train and validation data grouped by class id
    return train_in,val_in,Y_train_e,Y_val_e,nb_classes




def get_batch_random(batch_size,s="val"):
    """
    Create batch of APN triplets with a complete random strategy
    
    Arguments:
    batch_size -- integer 

    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,features)
    """
    if s == 'train':
        X = train_in
    else:
        X = val

    a,m = X[0].shape
    
    
    # initialize result
    triplets=[np.zeros((batch_size,m)) for i in range(3)]
    
    for i in range(batch_size):
        #Pick one random class for anchor
        anchor_class = np.random.randint(0, nb_classes)
        nb_sample_available_for_class_AP = X[anchor_class].shape[0]
        
        #Pick two different random pics for this class => A and P. You can use same anchor as P if there is one one element for anchor
        if nb_sample_available_for_class_AP<=1:
            continue
        [idx_A,idx_P] = np.random.choice(nb_sample_available_for_class_AP,size=2 ,replace=False)
        
        #Pick another class for N, different from anchor_class
        negative_class = (anchor_class + np.random.randint(1,nb_classes)) % nb_classes
        nb_sample_available_for_class_N = X[negative_class].shape[0]
        if nb_sample_available_for_class_N <=1:
            continue
        
        #Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)

        triplets[0][i,:] = X[anchor_class][idx_A,:]
        triplets[1][i,:] = X[anchor_class][idx_P,:]
        triplets[2][i,:] = X[negative_class][idx_N,:]

    return np.array(triplets)


def build_network(input_shape , embeddingsize):
    '''
    Define the neural network to learn image similarity
    Input : 
            input_shape : shape of input features
            embeddingsize : vectorsize used to encode our features 
    '''
   
    
    #in_ = Input(train.shape)
    net = Sequential()
    net.add(Dense(1024,  activation='tanh', input_shape=input_shape))
    net.add(Dense(512, activation='relu'))
    net.add(Dense(1024, activation='relu'))
    net.add(Dense(128, activation='tanh'))
    net.add(Dense(32, activation='sigmoid'))
    net.add(Dense(embeddingsize, activation= None))
     #Force the encoding to live on the d-dimentional hypershpere
    net.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))

    
    return net


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = 0.01
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
    


def build_model(input_shape, network, margin=0.01):
    '''
    Define the Keras Model for training 
        Input : 
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
    
    '''
     # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input") 
    
    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    
    #TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    
    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
    
    # return the model
    return network_train




def compute_dist(a,b):
    return np.sum(np.square(a-b))

def get_batch_hard(draw_batch_size,hard_batchs_size,norm_batchs_size,network,s="val"):
    """
    Create batch of APN "hard" triplets
    
    Arguments:
    draw_batch_size -- integer : number of initial randomly taken samples   
    hard_batchs_size -- interger : select the number of hardest samples to keep
    norm_batchs_size -- interger : number of random samples to add

    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,feature_length)
    """
    if s == 'train':
        X = train_in
    else:
        X = val

    #Step 1 : pick a random batch to study
    studybatch = get_batch_random(draw_batch_size,X)
            
            #Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))
            
            #Compute embeddings for anchors, positive and negatives
    A = network.predict(studybatch[0])
    P = network.predict(studybatch[1])
    N = network.predict(studybatch[2])
            
            #Compute d(A,P)-d(A,N)
    studybatchloss = np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1)
            
            #Sort by distance (high distance first) and take the 
    selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]
            
            #Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection),norm_batchs_size,replace=False)
            
    selection = np.append(selection,selection2)
            
    triplets = [studybatch[0][selection,:], studybatch[1][selection,:],studybatch[2][selection,:]]
#    trip_A = studybatch[0][selection,:]
#    trip_B = studybatch[1][selection,:]
#    trip_C = studybatch[2][selection,:]
    
        #trip_A=np.array(trip_A)
    return triplets
    
def get_dataset(draw_batch_size,hard_batchs_size,norm_batchs_size,network,steps_per_epoch):
    #generate the dataset of hard triplets for training by stacking them together.
    for x in range(steps_per_epoch):
        A,P,N = get_batch_hard(draw_batch_size,hard_batchs_size,norm_batchs_size,network,s="val")
        if x==0:
           hard_dataset_A = A
           hard_dataset_P = P
           hard_dataset_N = N
           continue
        hard_dataset_A =np.vstack((hard_dataset_A,A))
        hard_dataset_P =np.vstack((hard_dataset_P,P))
        hard_dataset_N =np.vstack((hard_dataset_N,N)) 
    return [hard_dataset_A,hard_dataset_P,hard_dataset_N]


def form_test_data(val):
    #draws data at random from the test data to test our model. it stacks sampled data using vstack
   #create empty list to store class labels
    Y = []
    #Add examples from each class to variable X
    for i in range(nb_classes):
        Y.append(i)
        #adds two examples if there are more than two examples per class
        if val[i].shape[0]>=2:
            dat_index = np.random.randint(val[i].shape[0],size=2)
            if dat_index.all() ==0:
                continue
            if i==0:
                X=val[i][dat_index[0]]
                X = np.vstack((X,val[i][dat_index[1]]))
                Y.append(i)
                continue
            #add only one example if there is only one example in that class
            X = np.vstack((X,val[i][dat_index[0]]))
            X = np.vstack((X,val[i][dat_index[1]]))
            Y.append(i)
        else:
            dat_index = np.random.randint(val[i].shape[0],size =1)
            if i==0:
                X=val[i][dat_index]
                continue
            X = np.vstack((X,val[i][dat_index]))
            
            
    return np.array(X),np.array(Y)     
            



# This function is used to find the distance between two different embeddings
def compute_probs(network,X,Y):
    '''
    Input
        network : current NN to compute embeddings
        X : tensor of shape (m,features) containing features to evaluate
        Y : tensor of shape (m,) containing true class
        
    Returns
        probs : array of shape (m,m) containing distances
    
    '''
    m = X.shape[0]
    nbevaluation = int(m*(m-1)/2)
    probs = np.zeros((nbevaluation))
    y = np.zeros((nbevaluation))
    same = np.zeros((m))
    diff = np.zeros((m))
    
    #Compute all embeddings for all pics with current network
    embeddings = network.predict(X[0:m])
    
    
    #size_embedding = embeddings.shape[1]
    
    #For each pics of our dataset
    k = 0
    for i in range(m):
            #Against all other images
            for j in range(i+1,m):
                #compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
                probs[k] = compute_dist(embeddings[i,:],embeddings[j,:])
                if (Y[i]==Y[j]):
                    y[k] = 1
                    #print("The distance between class {0} and class {1} is {2}.\tSAME".format(Y[i],Y[j],probs[k]))
                    #print('.....................................................................................') 
                    same[i]= probs[k]
                    
                else:
                    y[k] = 0
#                    print("The distance between class {0} and class {1} is {2}.\tDIFF".format(Y[i],Y[j],probs[k]))
#                    print('.....................................................................................')
                    diff[i]= probs[k]
                k += 1
    
    return probs,y,np.sort(same)[::-1],np.sort(diff)


if __name__ == "__main__":
    train_in,val,Y_train,Y_val,nb_classes = load_data()
    #input_shape = (train_in[0].shape[1],)
    input_shape = (val[0].shape[1],) 
    # create the base model    
    network = build_network(input_shape,embeddingsize=2)
    
    #create the dataset
    hard = get_dataset(100,32,32,network,1000)
    #create the siamese network
    network_train = build_model(input_shape,network)
#    #select optimiser and learning rate
    optimizer = Adam(lr = 0.00006)
#   
#    #compile the model
    network_train.compile(loss=None,optimizer=optimizer)
#    #start trainin the model
    t_start = time.time()
#    history = network_train.fit(hard,epochs=100 ,batch_size=32,verbose=2)
#    
    network.save('network_0.01.h5')
    
    
