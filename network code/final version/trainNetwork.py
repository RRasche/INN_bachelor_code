import tensorflow as tf
import numpy as np
import matplotlib.pyplot as py

from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow_probability import distributions as tfd
#from tensorflow.keras.callbacks import TensorBoard
#from time import time #for tensorboard

#own functions/classes
import gaussian #contains the function to create a gaussian mixture model
import losses #contains the MMD loss used to train the network
from networkClasses import INNBlock, INNBlockLast, INN #contains the classes to build an INN

#uncomment to run without using the GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def twoGauss(save = False):
    '''
    This function trains an INN on a Gaussain mixture model and then shows how well the network is able to generate data similar to the training data.
    Option:
        save <bool> if True the figures will be saved
    '''
    
    numb_of_dists = 2   # number of distributions in the gaussian mixture model
    size = 1000         # number of datapoints per distribution
    
    zero_noise=0.09    # scale of the Gaussian distribution used to fill the padding dimensions
    y_noise = 0.1       # scale of the Gaussian distribution used to perturb the label dimensions
    
    input_dim = 8       # input dimension
    latent_dim = 2      # latent dimension
    label_dim = 0       # label dimension
    
    # A Callback function is can be called during training to set the training parameters dynamically.
    class MyCallback(tf.keras.callbacks.Callback):
        def __init__(self, alpha, beta):
            '''
            Input:
                alpha, beta <Keras variable> placeholder for a loss weight
            '''
            self.alpha = alpha 
            self.beta = beta
            self.i = 0.0 # counts the number of training steps completed
        
        def on_train_begin(self, logs={}):
            ''' This function is called each time the training starts.'''
            self.i += 1.0
            
            #set values of alpha and beta
            self.alpha =K.set_value(self.alpha, 20* np.asarray([1., 2. * 0.003**(1.0 - (self.i / 30.0))]).min())
            self.beta = K.set_value(self.beta,np.asarray([self.i*1.5,16]).min())
    
    def genData(s,input_dim,latent_dim):
        '''
        Input:
            s           <tensorflow_probability distribution object> contains the distribution used to sample the latent variable.
            input_dim   <int> input dimension
            latent_dim  <int> latent dimension
        Output:
            x           <2D numpy array> contains x and y coorinates of the Gaussian mixture model
            z           <2D numpy array> contains the latent variables
        
        This function generates all the data needed to train the INN
        '''
        
        #get x and y coordinates as well as labels for a gaussian mixture model.
        ga = gaussian.mixture(numb_of_dists,size=size)
        
        #shuffle the data
        np.random.shuffle(ga)
        
        #store x and y coordinates
        x = np.delete(ga,2,axis=1)
        
        #sample a set of latent variables
        z = tf.dtypes.cast(s.sample(numb_of_dists*size),tf.dtypes.float32).numpy()

        return x,z
    
    #creating an INN
    inn = INN(4,5,latent_dim,label_dim,256,input_dim,2.0,[1,2,7,5,4,6,3,0],zero_noise,y_noise,0.1)
    
    #loading an already trained INN
    # inn.load_weights("./network/2Gauss.tf")

    #intilizing the callback variables
    alpha = K.variable(0.0)
    beta = K.variable(0.0)
    
    #compiling the model.
    inn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.9,amsgrad=False,clipvalue=15.0),loss=[losses.MMD_forward,'mse','mse',losses.MMD_backward],loss_weights=[100.0,4.0,4.0,100.0],metrics=['accuracy'])
    
    #creating distribution objects used to sample the latent variable (s) and padding dimension (s2)
    latent_mean = np.zeros(2)
    zero_mean = np.zeros(input_dim -2) 

    s = tfd.Sample(tfd.Normal(loc=latent_mean, scale=1.0))
    s2 = tfd.Sample(tfd.Normal(loc=zero_mean, scale=zero_noise))
    
    
    #training loop
    for i in range(60):
       
        #generate data
        x_pad = s2.sample((numb_of_dists*size)).numpy()
        x,z = genData(s,input_dim,latent_dim)
        
        #concatenate the data to their final form used in the training process
        x_t = np.concatenate((x,x_pad),axis=1)         
        y_t = np.concatenate([np.random.normal(loc=0.0,scale=zero_noise,size=(len(x),input_dim - label_dim - latent_dim))],axis=1)
        
        #training the network
        inn.fit(x_t,[z,y_t,x_t[:,:input_dim],x],epochs=2,batch_size=2000,verbose=2,callbacks=[MyCallback(alpha,beta)]) #,callbacks=[tensorboard]
    
    #saving the network weights
    # inn.save_weights('./network/2Gauss.tf')
    
    
    #generating new data to plot the results
    x_pad = s2.sample((numb_of_dists*size)).numpy().astype(np.float32)
    x,z = genData(s,input_dim,latent_dim)
    #z= np.random.uniform(low=-1.0,high=1.0,size=(2000,2))
    
    x_t = np.concatenate((x,x_pad),axis=1)
    y_t = np.concatenate([np.random.normal(loc=0.0,scale=zero_noise,size=(len(x),input_dim - label_dim - latent_dim))],axis=1)
    z_t = np.concatenate([z,y_t],axis=1)
    
    
    show_size = 2000 
    
    #using the generated data to perform an inverse pass through the network
    out = inn.inv(z_t[:show_size])  

    #splitting the output into x and y coordinates
    x_coord,y_coord = out[0].numpy().T
   
    fig=py.figure(figsize=(14,3.5))
    fig.subplots_adjust(left=0.1,right = 0.9,bottom=0.25,top=0.9)
    
    ax = fig.add_subplot(111)
    ax.set_xlabel("X",fontsize=25)
    ax.set_ylabel("Y",fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    #///////////////////////////////////
    # ax.set_title("Generated Data",fontsize=25)
    ax.set_ylim(bottom = -0.5,top =0.5)
    ax.set_xlim(left = -3.1,right=3.1)
    #//////////////////////////////
    ax.scatter(x_coord,y_coord,c='red',s=2.0)
    if save:
        fig.savefig('2gaussGen.png',dpi=300)
    
    fig=py.figure(figsize=(14,3.5))
    fig.subplots_adjust(left=0.1,right = 0.9,bottom=0.25,top=0.9)
    
    ax = fig.add_subplot(111)
    ax.set_xlabel("X",fontsize=25)
    ax.set_ylabel("Y",fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_ylim(bottom = -0.5,top =0.5)
    ax.set_xlim(left = -3.1,right=3.1)
    ax.scatter(x[:show_size,0],x[:show_size,1],c='red',s=2.0)
    if save:
        fig.savefig('2gaussOrig.png',dpi=300)

    py.show()
    
def fourGauss(save = False):
    '''
    This function trains an INN on a Gaussain mixture model and then shows how well the network is able to generate data similar to the training data.
    Option:
        save <bool> if True the figures will be saved
    '''
    
    numb_of_dists = 4   # number of distributions in the gaussian mixture model
    size = 1000         # number of datapoints per distribution
    
    zero_noise=0.06     # scale of the Gaussian distribution used to fill the padding dimensions
    y_noise = 0.1     # scale of the Gaussian distribution used to perturb the label dimensions
    
    input_dim = 10     # input dimension
    latent_dim = 2      # latent dimension
    label_dim = 2       # label dimension
    
    # A Callback function is can be called during training to set the training parameters dynamically.
    class MyCallback(tf.keras.callbacks.Callback):
        def __init__(self, alpha, beta):
            '''
            Input:
                alpha, beta <Keras variable> placeholder for a loss weight
            '''
                
            self.alpha = alpha 
            self.beta = beta
            self.i = 0.0 # counts the number of training steps completed
        
        def on_train_begin(self, logs={}):
            ''' This function is called each time the training starts.'''
            self.i += 1.0
            
            #set values of alpha and beta
            self.alpha =K.set_value(self.alpha, 850* np.asarray([1., 2. * 0.002**(1.0 - (self.i / 60.0))]).min())
            self.beta = K.set_value(self.beta,np.asarray([self.i*0.7,21]).min())
    
    def genData(s,input_dim,latent_dim,label_dim):
        '''
        Input:
            s           <tensorflow_probability distribution object> contains the distribution used to sample the latent variable.
            input_dim   <int> input dimension
            latent_dim  <int> latent dimension
            label_dim   <int> label dimension
        Output:
            x           <2D numpy array> contains x and y coorinates of the Gaussian mixture model
            y           <2D numpy array> contains the perturbed labels 
            z           <2D numpy array> contains the latent variables
            y_clean     <2D numpy array> contains the unperturb version of the labels
        
        This function generates all the data needed to train the INN
        '''
        
        #get x and y coordinates as well as labels for a gaussian mixture model.
        ga = gaussian.mixture(numb_of_dists,size=size)
        
        #shuffle the data
        np.random.shuffle(ga)
        
        #store x and y coordinates
        x = np.delete(ga,2,axis=1)
        
        #sample a set of latent variables
        z = tf.dtypes.cast(s.sample(numb_of_dists*size),tf.dtypes.float32).numpy()
        
        #get the labels
        _,_,y_load = ga.T
        
        #keep the labels in a custom form
        y_clean = np.zeros((numb_of_dists*size,label_dim))
        for j in range(len(y_clean)):
            if y_load[j] < 2:
                y_clean[j][0] += 1.0
            else:
                if y_load[j] < 4:
                    y_clean[j][1] += 1.0
        
        #perturb the labels
        y = y_clean + np.random.normal(loc=0.0,scale=y_noise,size=(len(y_load),label_dim))
                    
        return x,y,z,y_clean
    
    #creating an INN
    inn = INN(4,4,latent_dim,label_dim,256,input_dim,2.0,[2, 8, 9, 5, 6, 3, 7, 1, 4, 0],zero_noise,y_noise,0.1)
    
    #loading an already trained INN
    # inn.load_weights("./network/4Gauss.tf")

    #intilizing the callback variables
    alpha = K.variable(0.0)
    beta = K.variable(0.0)
    
    #compiling the model.
    inn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.7, beta_2=0.75,amsgrad=False,clipvalue=15.0),loss=[losses.MMD_forward,'mse','mse',losses.MMD_backward],loss_weights=[alpha,0.3,0.3,alpha],metrics=['accuracy'])
    
    #creating distribution objects used to sample the latent variable (s) and padding dimension (s2)
    latent_mean = np.zeros(2)
    zero_mean = np.zeros(input_dim -2) 

    s = tfd.Sample(tfd.Normal(loc=latent_mean, scale=1.0))
    s2 = tfd.Sample(tfd.Normal(loc=zero_mean, scale=zero_noise))
    
    
    #training loop
    for i in range(60):
       
        x_pad = s2.sample((numb_of_dists*size)).numpy()
       
        x,y,z,y_clean = genData(s,input_dim,latent_dim,label_dim)
        
        #concatenate the data to their final form used in the training process
        x_t = np.concatenate((x,x_pad,y_clean),axis=1)         
        y_t = np.concatenate([y,np.random.normal(loc=0.0,scale=zero_noise,size=(len(x),input_dim - label_dim - latent_dim))],axis=1)
        z_t = np.concatenate([z,y],axis=1)
        
        #training the network
        inn.fit(x_t,[z_t,y_t,x_t[:,:input_dim],x],epochs=1,batch_size=1600,verbose=2,callbacks=[MyCallback(alpha,beta)]) #,callbacks=[tensorboard]
    
    #saving the network weights
    inn.save_weights('./network/4Gauss.tf')
    
    
    #generating new data to plot the results
    x_pad = s2.sample((numb_of_dists*size)).numpy().astype(np.float32)
    x,y,z,_= genData(s,input_dim,latent_dim,label_dim)
    
    x_t = np.concatenate((x,x_pad),axis=1)
    y_t = np.concatenate([y,np.random.normal(loc=0.0,scale=zero_noise,size=(len(x),input_dim - label_dim - latent_dim))],axis=1)
    z_t = np.concatenate([z,y_t],axis=1)
    
    
    show_size = 2000 
    
    #using the generated data to perform an inverse pass through the network
    out = inn.inv(z_t[:show_size])  

    #splitting the output into x and y coordinates
    x_coord,y_coord = out[0].numpy().T
   
    fig=py.figure(figsize=(11,11))
    fig.subplots_adjust(left=0.1,right = 0.9,bottom=0.2,top=0.8)
    
    ax = fig.add_subplot(111)
    ax.set_xlabel("X",fontsize=25)
    ax.set_ylabel("Y",fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    #///////////////////////////////////
    # ax.set_title("Generated Data",fontsize=25)
    ax.set_ylim(bottom = -3.1,top =3.1)
    ax.set_xlim(left = -3.1,right=3.1)
    #//////////////////////////////
    ax.scatter(x_coord,y_coord,cmap='Spectral',c=np.argmax(y[:show_size],axis=1),s=2.0)
    if save:
        fig.savefig('8gaussGen.png',dpi=300)
    
    fig=py.figure(figsize=(11,11))
    fig.subplots_adjust(left=0.1,right = 0.9,bottom=0.2,top=0.8)
    
    ax = fig.add_subplot(111)
    ax.set_xlabel("X",fontsize=25)
    ax.set_ylabel("Y",fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_ylim(bottom = -3.1,top =3.1)
    ax.set_xlim(left = -3.1,right=3.1)
    ax.scatter(x[:show_size,0],x[:show_size,1],cmap='Spectral',c=np.argmax(y[:show_size],axis=1),s=2.0)
    if save:
        fig.savefig('8gaussOrig.png',dpi=300)

    py.show()
    
def eightGauss(save = False):
    '''
    This function trains an INN on a Gaussain mixture model and then shows how well the network is able to generate data similar to the training data.
    Option:
        save <bool> if True the figures will be saved
    '''
    
    numb_of_dists = 8   # number of distributions in the gaussian mixture model
    size = 1000         # number of datapoints per distribution
    
    zero_noise=0.08     # scale of the Gaussian distribution used to fill the padding dimensions
    y_noise = 0.105     # scale of the Gaussian distribution used to perturb the label dimensions
    
    input_dim = 16      # input dimension
    latent_dim = 2      # latent dimension
    label_dim = 4       # label dimension
    
    # A Callback function is can be called during training to set the training parameters dynamically.
    class MyCallback(tf.keras.callbacks.Callback):
        def __init__(self, alpha, beta):
            '''
            Input:
                alpha, beta <Keras variable> placeholder for a loss weight
            '''
                
            self.alpha = alpha 
            self.beta = beta
            self.i = 0.0 # counts the number of training steps completed
        
        def on_train_begin(self, logs={}):
            '''This function is called each time the training starts.'''
            self.i += 1.0
            
            #set values of alpha and beta
            self.alpha =K.set_value(self.alpha, 20* np.asarray([1., 2. * 0.003**(1.0 - (self.i / 30.0))]).min())
            self.beta = K.set_value(self.beta,np.asarray([self.i*3,21]).min())
    
    def genData(s,input_dim,latent_dim,label_dim):
        '''
        Input:
            s           <tensorflow_probability distribution object> contains the distribution used to sample the latent variable.
            input_dim   <int> input dimension
            latent_dim  <int> latent dimension
            label_dim   <int> label dimension
        Output:
            x           <2D numpy array> contains x and y coorinates of the Gaussian mixture model
            y           <2D numpy array> contains the perturbed labels 
            z           <2D numpy array> contains the latent variables
            y_clean     <2D numpy array> contains the unperturb version of the labels
        
        This function generates all the data needed to train the INN
        '''
        
        #get x and y coordinates as well as labels for a gaussian mixture model.
        ga = gaussian.mixture(numb_of_dists,size=size)
        
        #shuffle the data
        np.random.shuffle(ga)
        
        #store x and y coordinates
        x = np.delete(ga,2,axis=1)
        
        #sample a set of latent variables
        z = tf.dtypes.cast(s.sample(numb_of_dists*size),tf.dtypes.float32).numpy()
        
        #get the labels
        _,_,y_load = ga.T
        
        #keep the labels in a custom form
        y_clean = np.zeros((numb_of_dists*size,label_dim))
        for j in range(len(y_clean)):
            if y_load[j] < 4:
                y_clean[j][0] += 1.0
            else:
                if y_load[j] < 6:
                    y_clean[j][1] += 1.0
                else:
                    y_clean[j][int(y_load[j]-4)] += 1.0
        
        #perturb the labels
        y = y_clean + np.random.normal(loc=0.0,scale=y_noise,size=(len(y_load),label_dim))
                    
        return x,y,z,y_clean
    
    #creating an INN
    inn = INN(3,3,latent_dim,label_dim,384,input_dim,2.0,[7, 12, 5, 15, 2, 14, 6, 10, 8, 3, 1, 11, 13, 9, 4, 0],zero_noise,y_noise,0.05)
    
    #loading an already trained INN
    # inn.load_weights("./network/8Gauss.tf")

    #intilizing the callback variables
    alpha = K.variable(0.0)
    beta = K.variable(0.0)
    
    #compiling the model.
    inn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.5,amsgrad=False,clipvalue=15.0),loss=[losses.MMD_forward,'mse','mse',losses.MMD_backward],loss_weights=[18.0,0.08,0.08,beta],metrics=['accuracy'])
    
    #creating distribution objects used to sample the latent variable (s) and padding dimension (s2)
    latent_mean = np.zeros(2)
    zero_mean = np.zeros(input_dim -2) 

    s = tfd.Sample(tfd.Normal(loc=latent_mean, scale=1.0))
    s2 = tfd.Sample(tfd.Normal(loc=zero_mean, scale=zero_noise))
    
    
    #training loop
    for i in range(30):
       
        x_pad = s2.sample((numb_of_dists*size)).numpy()
       
        x,y,z,y_clean = genData(s,input_dim,latent_dim,label_dim)
        
        #concatenate the data to their final form used in the training process
        x_t = np.concatenate((x,x_pad,y_clean),axis=1)         
        y_t = np.concatenate([y,np.random.normal(loc=0.0,scale=zero_noise,size=(len(x),input_dim - label_dim - latent_dim))],axis=1)
        z_t = np.concatenate([z,y],axis=1)
        
        #training the network
        inn.fit(x_t,[z_t,y_t,x_t[:,:input_dim],x],epochs=1,batch_size=200,verbose=2,callbacks=[MyCallback(alpha,beta)]) #,callbacks=[tensorboard]
    
    #saving the network weights
    # inn.save_weights('./network/8Gauss.tf')
    
    
    #generating new data to plot the results
    x_pad = s2.sample((numb_of_dists*size)).numpy().astype(np.float32)
    x,y,z,_= genData(s,input_dim,latent_dim,label_dim)
    
    x_t = np.concatenate((x,x_pad),axis=1)
    y_t = np.concatenate([y,np.random.normal(loc=0.0,scale=zero_noise,size=(len(x),input_dim - label_dim - latent_dim))],axis=1)
    z_t = np.concatenate([z,y_t],axis=1)
    
    
    show_size = 2000 
    
    #using the generated data to perform an inverse pass through the network
    out = inn.inv(z_t[:show_size])  

    #splitting the output into x and y coordinates
    x_coord,y_coord = out[0].numpy().T
   
    fig=py.figure(figsize=(11,11))
    fig.subplots_adjust(left=0.1,right = 0.9,bottom=0.2,top=0.8)
    
    ax = fig.add_subplot(111)
    ax.set_xlabel("X",fontsize=25)
    ax.set_ylabel("Y",fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    #///////////////////////////////////
    # ax.set_title("Generated Data",fontsize=25)
    ax.set_ylim(bottom = -3.1,top =3.1)
    ax.set_xlim(left = -3.1,right=3.1)
    #//////////////////////////////
    ax.scatter(x_coord,y_coord,cmap='Spectral',c=np.argmax(y[:show_size],axis=1),s=2.0)
    if save:
        fig.savefig('8gaussGen.png',dpi=300)
    
    fig=py.figure(figsize=(11,11))
    fig.subplots_adjust(left=0.1,right = 0.9,bottom=0.2,top=0.8)
    
    ax = fig.add_subplot(111)
    ax.set_xlabel("X",fontsize=25)
    ax.set_ylabel("Y",fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_ylim(bottom = -3.1,top =3.1)
    ax.set_xlim(left = -3.1,right=3.1)
    ax.scatter(x[:show_size,0],x[:show_size,1],cmap='Spectral',c=np.argmax(y[:show_size],axis=1),s=2.0)
    if save:
        fig.savefig('8gaussOrig.png',dpi=300)

    py.show()

#Tain an INN on Gaussian mixture model with either two, four or eight modes.
#If save is set to True, it will save the resulting inverse plot as a file.
twoGauss(save=True)
#fourGauss(save=True)
# eightGauss(save=True)
