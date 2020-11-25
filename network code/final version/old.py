import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
import numpy as np

class INNBlockOld(tf.keras.layers.Layer):
    def __init__(self,hdLayers,layerSize, input_dim, clipval,permutation,**kwargs):
        # Input: 
            # hdLayers      <int> defines the amount of hidden layers per sub network
            # layerSize     <int> defines the amount of neurons in each hidden layer
            # input_dim     <int> defines the dimension of the input
            # clipval       <float> defines the maximum argument in the exponential function
            # permutation   <int array> defines a permutation to shuffle data at the end of a block
        #
        # This class contains all the data and function needed for the smallest unit of an invertible neural network.
        # A diagram can be seen in ../graphics/INN/innForward.png
          
        #initilizing variables
        super(INNBlockOld, self).__init__(**kwargs)
        self.size = input_dim//2
        self.clipvalue = clipval
        max_norm_value = 2.0
        self.pi = 3.1415926
        self.permutation = permutation
        self.invPermutation = np.argsort(permutation)

        #initilizing the sub networks of the INN block in a list
        self.networks = []
        for curLayer in range(4):
            self.networks.append([])
            # the first layer in a subnetwork needs to have the correct input_shape. 
            #The 'None' in input_shape is just a placeholder, that will be filled with the batch_size while training the network.
            self.networks[curLayer].append(layers.Dense(layerSize,activation=tf.nn.relu,input_shape=(None,self.size), bias_initializer=tf.constant_initializer(0.0),kernel_initializer = tf.keras.initializers.GlorotNormal(), kernel_constraint=tf.keras.constraints.max_norm(max_norm_value)))
            
            for j in range(1,hdLayers -1):
                self.networks[curLayer].append(layers.Dense(layerSize,activation=tf.nn.relu,input_shape=(None,layerSize), bias_initializer=tf.constant_initializer(0.0),kernel_initializer = tf.keras.initializers.GlorotNormal(), kernel_constraint=tf.keras.constraints.max_norm(max_norm_value)))
            
            #the last layer needs to have the correct number of neurons so that the output has the same dimension as the input that originally entered the network.
            self.networks[curLayer].append(layers.Dense(self.size,activation=tf.keras.activations.linear,input_shape=(None,layerSize), bias_initializer=tf.constant_initializer(0.0),kernel_initializer = tf.keras.initializers.GlorotNormal(), kernel_constraint=tf.keras.constraints.max_norm(max_norm_value)))
                

    def call(self, inputs):
        # Input:
        #   inputs <2D tensorflow tensor> contains the training data
        # Output:
        #   out <2D tensorflow tensor> contains the shuffled output of the block.
        # 
        # This function contains the forward pass through the block. 
        
        
        # splitting the data
        input1 = inputs[:,:self.size]
        input2 = inputs[:,self.size:]
        
        x2 = input2
        
        # this for loop applies subnetwork 0 to the data in x2.
        for layer in self.networks[0]:
            x2 = layer(x2)
            
        # linking the data together
        x2 = tf.math.exp(tf.keras.backend.clip(x2,-self.clipvalue,self.clipvalue))
        v1_temp = tf.math.multiply(x2,input1)
        
        x2 = input2
        for layer in self.networks[1]:
            x2 = layer(x2)
        v1 = tf.math.add(v1_temp,x2)

        x1 = v1
        for layer in self.networks[2]:
            x1 = layer(x1)   
        x1 = tf.math.exp(tf.keras.backend.clip(x1,-self.clipvalue,self.clipvalue))
        v2_temp = tf.math.multiply(x1,input2)
        
        x1 = v1
        for layer in self.networks[3]:
            x1 = layer(x1)
        v2 = tf.math.add(v2_temp,x1)
        out = tf.concat([v1,v2],1)
        
        # the data is permuted
        out = tf.gather(out,self.permutation,axis=1)
        return out
        
    def inverse_call(self, inputs):
        # This function is the inverse to the function call() above. It contains the inverse pass through the block.
        # Input:
        #   inputs <2D tensorflow tensor> contains the inverse data. Output of a previous inverse pass through another block.
        # Output:
        #   out <2D tensorflow tensor> contains the output of the block.
        
        #unshuffle the data and split it.
        inp = tf.gather(inputs, self.invPermutation,axis=1)
        v1 = inp[:,:self.size]
        v2 = inp[:,self.size:]
        
        x1 = v1
        for layer in self.networks[3]:
            x1 = layer(x1)
        u2_temp = tf.math.subtract(v2,x1)
        
        x1 = v1
        for layer in self.networks[2]:
            x1 = layer(x1)   
        x1 = tf.math.exp(-tf.keras.backend.clip(x1,-self.clipvalue,self.clipvalue))
        u2 = tf.math.multiply(u2_temp,x1)
        
        x2 = u2
        for layer in self.networks[1]:
            x2 = layer(x2)
        u1_temp = tf.math.subtract(v1,x2)
        
        for layer in self.networks[0]:
            x2 = layer(x2)
        x2 = tf.math.exp(-tf.keras.backend.clip(x2,-self.clipvalue,self.clipvalue))
        u1 = tf.math.multiply(u1_temp,x2)
        
        out = tf.concat([u1,u2],1)

        return out

class INNBlockLastOld(tf.keras.layers.Layer):
    def __init__(self,hdLayers,layerSize, input_dim, clipval,**kwargs):
        # Input: 
            # hdLayers      <int> defines the amount of hidden layers per sub network
            # layerSize     <int> defines the amount of neurons in each hidden layer
            # input_dim     <int> defines the dimension of the input
            # clipval       <float> defines the maximum argument in the exponential function
            # permutation   <int array> defines a permutation to shuffle data at the end of a block
        #
        # This class contains all the data and function needed for the smallest unit of an invertible neural network.
        # A diagram can be seen in ../graphics/INN/innForward.png
          
        #initilizing variables
        super(INNBlockLastOld, self).__init__(**kwargs)
        self.size = input_dim//2
        self.clipvalue = clipval
        max_norm_value = 2.0
        self.pi = 3.1415926
        #initilizing the sub networks of the INN block in a list
        self.networks = []
        for curLayer in range(4):
            self.networks.append([])
            # the first layer in a subnetwork needs to have the correct input_shape. 
            #The 'None' in input_shape is just a placeholder, that will be filled with the batch_size while training the network.
            self.networks[curLayer].append(layers.Dense(layerSize,activation=tf.nn.relu,input_shape=(None,self.size), bias_initializer=tf.constant_initializer(0.0),kernel_initializer = tf.keras.initializers.GlorotNormal(), kernel_constraint=tf.keras.constraints.max_norm(max_norm_value)))
            
            for j in range(1,hdLayers -1):
                self.networks[curLayer].append(layers.Dense(layerSize,activation=tf.nn.relu,input_shape=(None,layerSize), bias_initializer=tf.constant_initializer(0.0),kernel_initializer = tf.keras.initializers.GlorotNormal(), kernel_constraint=tf.keras.constraints.max_norm(max_norm_value)))
            
            #the last layer needs to have the correct number of neurons so that the output has the same dimension as the input that originally entered the network.
            self.networks[curLayer].append(layers.Dense(self.size,activation=tf.keras.activations.linear,input_shape=(None,layerSize), bias_initializer=tf.constant_initializer(0.0),kernel_initializer = tf.keras.initializers.GlorotNormal(), kernel_constraint=tf.keras.constraints.max_norm(max_norm_value)))
                

    def call(self, inputs):
        # Input:
        #   inputs <2D tensorflow tensor> contains the training data
        # Output:
        #   out <2D tensorflow tensor> contains the shuffled output of the block.
        # 
        # This function contains the forward pass through the block. 
        
        
        # splitting the data
        input1 = inputs[:,:self.size]
        input2 = inputs[:,self.size:]
        
        x2 = input2
        
        # this for loop applies subnetwork 0 to the data in x2.
        for layer in self.networks[0]:
            x2 = layer(x2)
            
        # linking the data together
        x2 = tf.math.exp(tf.keras.backend.clip(x2,-self.clipvalue,self.clipvalue))
        v1_temp = tf.math.multiply(x2,input1)
        
        x2 = input2
        for layer in self.networks[1]:
            x2 = layer(x2)
        v1 = tf.math.add(v1_temp,x2)

        x1 = v1
        for layer in self.networks[2]:
            x1 = layer(x1)   
        x1 = tf.math.exp(tf.keras.backend.clip(x1,-self.clipvalue,self.clipvalue))
        v2_temp = tf.math.multiply(x1,input2)
        
        x1 = v1
        for layer in self.networks[3]:
            x1 = layer(x1)
        v2 = tf.math.add(v2_temp,x1)
        out = tf.concat([v1,v2],1)
        
        return out
        
    def inverse_call(self, inputs):
        # This function is the inverse to the function call() above. It contains the inverse pass through the block.
        # Input:
        #   inputs <2D tensorflow tensor> contains the inverse data. Output of a previous inverse pass through another block.
        # Output:
        #   out <2D tensorflow tensor> contains the output of the block.
        
        #unshuffle the data and split it.
        v1 = inputs[:,:self.size]
        v2 = inputs[:,self.size:]
        
        x1 = v1
        for layer in self.networks[3]:
            x1 = layer(x1)
        u2_temp = tf.math.subtract(v2,x1)
        
        x1 = v1
        for layer in self.networks[2]:
            x1 = layer(x1)   
        x1 = tf.math.exp(-tf.keras.backend.clip(x1,-self.clipvalue,self.clipvalue))
        u2 = tf.math.multiply(u2_temp,x1)
        
        x2 = u2
        for layer in self.networks[1]:
            x2 = layer(x2)
        u1_temp = tf.math.subtract(v1,x2)
        
        for layer in self.networks[0]:
            x2 = layer(x2)
        x2 = tf.math.exp(-tf.keras.backend.clip(x2,-self.clipvalue,self.clipvalue))
        u1 = tf.math.multiply(u1_temp,x2)
        
        out = tf.concat([u1,u2],1)

        return out

class INNOld(tf.keras.Model):
    def __init__(self,blocks,hdLayers,latent_dim,label_dim, layerSize,input_dim,clipval,permutation,zero_noise,y_noise,latent_perturb_scale,**kwargs):
        # Input
        #   blocks      <int> defines the amount of blocks that the INN should contain
        #   hdLayers    <int> defines the amount of hidden layers per sub network
        #   latent_dim  <int> defines the dimension of the latent space
        #   layerSize   <int> defines the amount of neurons in each hidden layer
        #   input_dim   <int> defines the dimension of the input
        #   clipval     <float> defines the maximum argument in the exponential function
        #   permutation <int array> defines a permutation to shuffle data at the end of a block
        #   zero_noise  <float> scale of Gaussian distribution used to fill the padding dimensions
        #   y_noise     <float> scale of Gaussian distribution used to perturb the labels.
        #   latent_perturb_scale <float> defines how much the latent variable will be perturbed in the call function
        #
        # This class contains a whole INN with multiple blocks.
        
        super(INNOld, self).__init__(**kwargs)
        
        #initilizing variables
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.latent_perturb_scale = latent_perturb_scale
        
        #initilizing the random distributions that are needed
        latent_mean = tf.zeros(2)
        latent_scale = tf.ones(2)  
        zero_mean = tf.zeros(self.input_dim-2)
        label_mean = tf.zeros(self.label_dim)
        
        self.s = tfd.Sample(tfd.Normal(loc=latent_mean, scale=latent_scale))
        self.s2 = tfd.Sample(tfd.Normal(loc=zero_mean, scale=zero_noise))
        self.s3 = tfd.Sample(tfd.Normal(loc=label_mean,scale=y_noise))
        
        #filling a list with the INNBlocks for the network
        self.blocks = []
        for curBlock in range(blocks-1):
            self.blocks.append(INNBlockOld(hdLayers,layerSize,input_dim,clipval,permutation))
            
        self.blocks.append(INNBlockLastOld(hdLayers,layerSize,input_dim,clipval))
    
    def call(self, inputs):
        # Input:
        #   inputs <2D tensorflow tensor> contains the training data and the clean labels.
        # Output:
        #   latents_and_labels      <2D tensorflow tensor> contains the output of the INN for the latent variable as well as the labels
        #   labels_and_padding      <2D tensorflow tensor> contains the output of the INN for the labels and the padding dimensions.
        #   inverse_perturbed       <2D tensorflow tensor> contains the full inverse output of the network obtained by using a slightly 
        #                                                  perturbed latent variable from the network output
        #   inverse_newly_sampled   <2D tensorflow tensor> contains only the x and y coordinates obtained by sampling a new set of latent variables.
        #
        # This function contains a forward and two inverse passes through the network used to train the network. 
        
        #splitting the input into the training data and the labels
        x = inputs[:,:self.input_dim]
        labels = inputs[:,self.input_dim:]
        
        
        #forward pass through the INN
        for block in self.blocks:
            x = block(x)
            
        #splitting of the latent variable
        latent = x[:,:self.latent_dim]
        
        #slightly perturb it
        latent_perturb = latent + self.latent_perturb_scale*tf.cast(self.s.sample(tf.shape(x)[0]),tf.dtypes.float32)
        
        #concatenate it with perturbed labels and padding dimensions
        backIn = tf.concat([latent_perturb,labels+tf.cast(self.s3.sample(tf.shape(x)[0]),tf.dtypes.float32),tf.cast(self.s2.sample(tf.shape(x)[0])[:,:self.input_dim - (self.latent_dim + self.label_dim)],tf.dtypes.float32)],axis=1) 
        
        #calculating an inverse pass
        y = backIn
        for block in reversed(self.blocks):
            y = block.inverse_call(y)
        inverse_perturbed = y 
        
        #sampling a new set of latent variables
        latent = tf.cast(self.s.sample(tf.shape(x)[0]),tf.dtypes.float32)
        
        #concatenating it with the perturb labels and padding dimensions
        backIn = tf.concat([latent,labels+tf.cast(self.s3.sample(tf.shape(x)[0]),tf.dtypes.float32),tf.cast(self.s2.sample(tf.shape(x)[0])[:,:self.input_dim - (self.latent_dim + self.label_dim)],tf.dtypes.float32)],axis=1)
        
        #inverse pass through the network
        y = backIn
        for block in reversed(self.blocks):
            y = block.inverse_call(y)
        inverse_newly_sampled = y[:,:self.latent_dim]
        
        latents_and_labels = x[:,:self.latent_dim+self.label_dim]
        labels_and_padding = x[:,self.latent_dim:]
        
        return [latents_and_labels,labels_and_padding,inverse_perturbed,inverse_newly_sampled]
        
    def inv(self,inputs):
        # Input:
        #   inputs <2D tensorflow tensor> contains a set of latent variables, labels and padding
        # Output:
        #   y[:,:self.latent_dim] -> latent varibles
        #   y[:,self.latent_dim:] -> labels + padding
        #
        # This function calculates an inverse pass through the INN.
        y = inputs
        for block in reversed(self.blocks):
            y =block.inverse_call(y)
        
        return [y[:,:self.latent_dim],y[:,self.latent_dim:]]
 