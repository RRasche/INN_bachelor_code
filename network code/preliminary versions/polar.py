import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as py
from tensorflow.keras.callbacks import TensorBoard
from time import time #for tensorboard
import gaussian

 

class INNBlock(tf.keras.layers.Layer):
    def __init__(self, layerSize, input_dim, min_clip,max_clip,**kwargs):
        super(INNBlock, self).__init__(name='innblock',**kwargs)
        self.size = input_dim//2
        self.layerSize = layerSize
        self.min_clip = min_clip
        self.max_clip = max_clip
        
    
        
        self.denseL11 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size), bias_initializer=tf.constant_initializer(0.01))
        self.denseL12 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL13 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL14 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL15 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL16 = layers.Dense(self.size,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        
        self.denseL21 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size), bias_initializer=tf.constant_initializer(0.01))
        self.denseL22 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL23 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL24 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL25 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL26 = layers.Dense(self.size,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        
        self.denseL31 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size), bias_initializer=tf.constant_initializer(0.01))
        self.denseL32 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL33 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL34 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL35 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL36 = layers.Dense(self.size,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        
        self.denseL41 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size), bias_initializer=tf.constant_initializer(0.01))
        self.denseL42 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL43 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL44 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL45 = layers.Dense(self.layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        self.denseL46 = layers.Dense(self.size,activation=tf.nn.leaky_relu,input_shape=(None,self.layerSize), bias_initializer=tf.constant_initializer(0.01))
        
        

    def call(self, inputs):
       # print(inputs)
        input1 = inputs[:,:self.size]#[:,:self.size]
        input2 = inputs[:,self.size:]
        
        x2 = self.denseL11(input2)
        x2 = self.denseL12(x2)
        x2 = self.denseL13(x2)
        x2 = self.denseL14(x2)
        x2 = self.denseL15(x2)
        x2 = self.denseL16(x2)
        x2 = tf.clip_by_value(x2,self.min_clip,self.max_clip)
        x2 = tf.math.exp(x2)
        v1_temp = tf.math.multiply(x2,input1)

        x2 = self.denseL21(input2)
        x2 = self.denseL22(x2)
        x2 = self.denseL23(x2)
        x2 = self.denseL24(x2)
        x2 = self.denseL25(x2)
        x2 = self.denseL26(x2)
        v1 = tf.math.add(v1_temp,x2)        

        x1 = self.denseL31(v1)
        x1 = self.denseL32(x1)
        x1 = self.denseL33(x1)
        x1 = self.denseL34(x1)
        x1 = self.denseL35(x1)
        x1 = self.denseL36(x1)
        x1 = tf.clip_by_value(x1,self.min_clip,self.max_clip)
        x1 = tf.math.exp(x1)
        v2_temp = tf.math.multiply(x1,input2)#inputs[:,1::2]
        
        x1 = self.denseL41(v1)
        x1 = self.denseL42(x1)
        x1 = self.denseL43(x1)
        x1 = self.denseL44(x1)
        x1 = self.denseL45(x1)
        x1 = self.denseL46(x1)
        v2 = tf.math.add(v2_temp,x1)
        out = tf.concat([v1,v2],1)
        # out = tf.gather(out,[3,0,9,5,2,6,4,1,7,8],axis=1)
        # out = tf.gather(out,[3,6,0,1,7,2,5,4],axis=1)
        return out
        
    def inverse_call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # inp = tf.gather(inputs,[1,7,4,0,6,3,5,8,9,2],axis=1)
        # inp = tf.gather(inputs,[2,3,5,0,7,6,1,4],axis=1)
        v1 = inputs[:,:self.size]
        v2 = inputs[:,self.size:]

        
        x1 = self.denseL41(v1)
        x1 = self.denseL42(x1)
        x1 = self.denseL43(x1)
        x1 = self.denseL44(x1)
        x1 = self.denseL45(x1)
        x1 = self.denseL46(x1)
        u2_temp = tf.math.subtract(v2,x1)

        x1 = self.denseL31(v1)
        x1 = self.denseL32(x1)
        x1 = self.denseL33(x1)
        x1 = self.denseL34(x1)
        x1 = self.denseL35(x1)
        x1 = self.denseL36(x1)
        x1 = tf.clip_by_value(x1,self.min_clip,self.max_clip)
        x1 = tf.math.exp(-x1)

        u2 = tf.math.multiply(u2_temp,x1)


        x2 = self.denseL21(u2)
        x2 = self.denseL22(x2)
        x2 = self.denseL23(x2)
        x2 = self.denseL24(x2)
        x2 = self.denseL25(x2)
        x2 = self.denseL26(x2)
        u1_temp = tf.math.subtract(v1,x2)

        x2 = self.denseL11(u2)
        x2 = self.denseL12(x2)
        x2 = self.denseL13(x2)
        x2 = self.denseL14(x2)
        x2 = self.denseL15(x2)
        x2 = self.denseL16(x2)
        x2 = tf.clip_by_value(x2,self.min_clip,self.max_clip)
        x2 = tf.math.exp(-x2)
        u1 = tf.math.multiply(u1_temp,x2)

        #out = tf.stack([u1,u2],axis=2)
        #out = tf.reshape(out,(batch_size,self.size*2))
        out = tf.concat([u1,u2],1)

        return out

class INNBlockLast(tf.keras.layers.Layer):
    def __init__(self, layerSize, input_dim, min_clip,max_clip,**kwargs):
        super(INNBlockLast, self).__init__(name='innblocklast',**kwargs)
        self.size = input_dim//2
        self.min_clip = min_clip
        self.max_clip = max_clip
    
        
        self.denseL11 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL12 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL13 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL14 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL15 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL16 = layers.Dense(self.size,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        
        self.denseL21 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL22 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL23 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL24 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL25 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL26 = layers.Dense(self.size,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        
        self.denseL31 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL32 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL33 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL34 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL35 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL36 = layers.Dense(self.size,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        
        self.denseL41 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL42 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL43 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL44 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL45 = layers.Dense(layerSize,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        self.denseL46 = layers.Dense(self.size,activation=tf.nn.leaky_relu,input_shape=(None,self.size),bias_initializer=tf.constant_initializer(0.01))
        
        

    def call(self, inputs):
       # print(inputs)
        input1 = inputs[:,:self.size]#[:,:self.size]
        input2 = inputs[:,self.size:]
        

        x2 = self.denseL11(input2)
        x2 = self.denseL12(x2)
        x2 = self.denseL13(x2)
        x2 = self.denseL14(x2)
        x2 = self.denseL15(x2)
        x2 = self.denseL16(x2)
        x2 = tf.clip_by_value(x2,self.min_clip,self.max_clip)
        x2 = tf.math.exp(x2)
        v1_temp = tf.math.multiply(x2,input1)

        x2 = self.denseL21(input2)
        x2 = self.denseL22(x2)
        x2 = self.denseL23(x2)
        x2 = self.denseL24(x2)
        x2 = self.denseL25(x2)
        x2 = self.denseL26(x2)
        v1 = tf.math.add(v1_temp,x2)        

        x1 = self.denseL31(v1)
        x1 = self.denseL32(x1)
        x1 = self.denseL33(x1)
        x1 = self.denseL34(x1)
        x1 = self.denseL35(x1)
        x1 = self.denseL36(x1)
        x1 = tf.clip_by_value(x1,self.min_clip,self.max_clip)
        x1 = tf.math.exp(x1)
        v2_temp = tf.math.multiply(x1,input2)#inputs[:,1::2]
        
        x1 = self.denseL41(v1)
        x1 = self.denseL42(x1)
        x1 = self.denseL43(x1)
        x1 = self.denseL44(x1)
        x1 = self.denseL45(x1)
        x1 = self.denseL46(x1)
        v2 = tf.math.add(v2_temp,x1)
        out = tf.concat([v1,v2],1)
        #out = tf.gather(out,[8,3,5,1,2,9,0,6,4,7],axis=1)
        return out
        
    def inverse_call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        #inp = tf.gather(inputs,[6,3,4,1,8,2,7,9,0,5],axis=1)
        v1 = inputs[:,:self.size]
        v2 = inputs[:,self.size:]
        
        x1 = self.denseL41(v1)
        x1 = self.denseL42(x1)
        x1 = self.denseL43(x1)
        x1 = self.denseL44(x1)
        x1 = self.denseL45(x1)
        x1 = self.denseL46(x1)
        u2_temp = tf.math.subtract(v2,x1)

        x1 = self.denseL31(v1)
        x1 = self.denseL32(x1)
        x1 = self.denseL33(x1)
        x1 = self.denseL34(x1)
        x1 = self.denseL35(x1)
        x1 = self.denseL36(x1)
        x1 = tf.clip_by_value(x1,self.min_clip,self.max_clip)
        x1 = tf.math.exp(-x1)
        u2 = tf.math.multiply(u2_temp,x1)


        x2 = self.denseL21(u2)
        x2 = self.denseL22(x2)
        x2 = self.denseL23(x2)
        x2 = self.denseL24(x2)
        x2 = self.denseL25(x2)
        x2 = self.denseL26(x2)
        u1_temp = tf.math.subtract(v1,x2)

        x2 = self.denseL11(u2)
        x2 = self.denseL12(x2)
        x2 = self.denseL13(x2)
        x2 = self.denseL14(x2)
        x2 = self.denseL15(x2)
        x2 = self.denseL16(x2)
        x2 = tf.clip_by_value(x2,self.min_clip,self.max_clip)
        x2 = tf.math.exp(-x2)
        u1 = tf.math.multiply(u1_temp,x2)
        
        #out = tf.stack([u1,u2],axis=2)
        #out = tf.reshape(out,(batch_size,self.size*2))
        out = tf.concat([u1,u2],1)
        return out

class INN(tf.keras.Model):
    def __init__(self,kenel_size,input_dim,min_clip,max_clip,**kwargs):
        super(INN, self).__init__(name='INN',**kwargs)
        self.block_1 = INNBlock(kenel_size,input_dim,min_clip,max_clip)
        self.block_2 = INNBlock(kenel_size,input_dim,min_clip,max_clip)
        self.block_3 = INNBlock(kenel_size,input_dim,min_clip,max_clip)
        self.block_4 = INNBlock(kenel_size,input_dim,min_clip,max_clip)
        self.block_5 = INNBlock(kenel_size,input_dim,min_clip,max_clip)

        # loc = np.zeros(2)
        # scale = np.ones(2) 
        # self.s = tfd.Sample(tfd.Normal(loc=loc, scale=scale))

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        # x = self.block_6(x)
        
        # backIn = tf.cast(self.s.sample(tf.shape(x)[0]),tf.dtypes.float32)
        # backIn = x + backIn
        # backIn = tf.concat([x[:,:8],backIn],axis=1)
        
        # y = self.block_6.inverse_call(backIn)
        # y = self.block_5.inverse_call(y)
        # y = self.block_4.inverse_call(y)
        # y = self.block_3.inverse_call(y)
        # y = self.block_2.inverse_call(y)
        # y = self.block_1.inverse_call(y)
        
        # y1 = y[:,:2]
        # y2 = y[:,2:]
        

        return x

    def inv(self,inputs):
        # x = self.block_1(inputs)
        # x = self.block_2(x)
        # x = self.block_3(x)
        # x = self.block_4(x)
        # x = self.block_5(x)
        # x = self.block_6(x)
        
    
        y = self.block_5.inverse_call(inputs)
        y = self.block_4.inverse_call(y)
        y = self.block_3.inverse_call(y)
        y = self.block_2.inverse_call(y)
        y = self.block_1.inverse_call(y)
        
        return y
    
    
class simpleModel(tf.keras.Model):
    def __init__(self):
        super(simpleModel,self).__init__()
        self.dense1 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,input_shape=(None,2))
        self.dense2 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,input_shape=(None,128))
        self.dense3 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,input_shape=(None,128))
        self.dense4 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,input_shape=(None,128))
        self.dense5 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,input_shape=(None,128))
        self.dense6 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,input_shape=(None,128))
        self.dense7 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,input_shape=(None,128))
        self.dense8 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,input_shape=(None,128))
        self.dense9 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,input_shape=(None,128))
        self.dense13 = tf.keras.layers.Dense(8,activation=tf.keras.activations.linear,input_shape=(None,128))
        self.dense14 = tf.keras.layers.Dense(2,activation=tf.keras.activations.linear,input_shape=(None,128))
        
        
    def call(self,inputs):
        x=self.dense1(inputs)
        x=self.dense2(x)
        x=self.dense3(x)
        x=self.dense4(x)
        x=self.dense5(x)
        x=self.dense6(x)
        x=self.dense7(x)
        x=self.dense8(x)
        x=self.dense9(x)
        x1=self.dense13(x)
        x2=self.dense14(x)
        return [x1,x2]

def polar(x,y):
    r= np.sqrt(np.power(x,2) + np.power(y,2))
    phi = np.arctan(y/x)
    return r,phi

#code for reproducing polar1, see ./confs-Inversemap.txt
def polar1(save=False):
        
    input_dim = 2
    inn = INN(64,input_dim,-1.0,1.0)
    inn.load_weights("./network/polar1.tf")

    inn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.99, beta_2=0.999, amsgrad=False),loss='mse',metrics=['accuracy'])

    loc1= np.asarray([3.0,0.0])
    loc2 = np.zeros(6)
    s = tfd.Sample(tfd.Uniform(low=[0.1,-5.0], high=[5.0,5.0]))
    s2 = tfd.Sample(tfd.Normal(loc=loc2, scale=0.05))

    for i in range(0):
        x,y = s.sample(10000).numpy().T
        x_t = np.asarray([x,y]).T
        r,phi = polar(x,y)
        y_t = np.asarray([r,phi]).T
        inn.fit(x_t,y_t,epochs=2,batch_size=256,verbose=2)


    
    x,y = np.random.normal(4.0,1.0,(600)),np.random.normal(0.0,1.0,(600))
    x_t = np.asarray([x,y]).T
    r,phi = polar(x,y)
    
    x_t = np.asarray([x,y]).T   #network input
    rp,phip = inn.predict(x_t).T
    y_t = np.asarray([r,phi]).T #inverse network input
    xp,yp = inn.inv(y_t).numpy().T
    
 
    # print(np.mean(np.abs(r-rr)))
    # print(np.mean(np.abs(phi-rphi)))
    
    # for polar1forward------------------------------------------------------------
    fig = py.figure(figsize=(11,9))
    fig.subplots_adjust(left=0.2,right = 0.95,bottom=0.2,top=0.95)
    
    ax = fig.add_subplot(111)
    ax.plot(rp*np.cos(phip),rp*np.sin(phip),'ro',label ='transformed')
    ax.plot(x,y,'go',label='original')
    
    ax.set_xlabel("X",fontsize=30)
    ax.set_ylabel("Y",fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=29)
    ax.legend(fontsize = 25)
    if save:
        fig.savefig('polar1forward.png',dpi=300)
    
    # for polar1backward - ------------------------------------------------------    fig = py.figure(figsize=(10,9))
    fig = py.figure(figsize=(11,9))
    fig.subplots_adjust(left=0.2,right = 0.95,bottom=0.2,top=0.95)
    
    ax = fig.add_subplot(111)
    ax.plot(xp,yp,'ro',label ='inverse transformed')
    ax.plot(x,y,'go',label='original')
    
    ax.set_xlabel("X",fontsize=30)
    ax.set_ylabel("Y",fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=29)
    ax.legend(fontsize = 25)
    if save:
        fig.savefig('polar1backward.png',dpi=300)
    # py.show()
    
    # for polar1mesh-----------------------------------------
    # rr, rphi = polar(xmesh,ymesh)
    # out = inn.predict(mesh)
    # r,phi = out.T
    # y_t = np.asarray([rr,rphi]).T
    # out = inn.inv(y_t).numpy()
    # xr,yr = out.T
    
    # fig = py.figure(figsize=(15,15))
    # fig.subplots_adjust(left=0.1,right = 0.9,bottom=0.2,top=0.8)
    
    # ax = fig.add_subplot(111)
    # ax.plot(r*np.cos(phi),r*np.sin(phi),'bo',label ='transformed')
    # ax.plot(xr,yr,'ro',label='inverse transform')
    # ax.plot(xmesh,ymesh,'go',label='original',alpha=0.5)

    # ax.set_xlabel("X",fontsize=25)
    # ax.set_ylabel("Y",fontsize=25)
    # ax.tick_params(axis='both', which='major', labelsize=24)
    # ax.legend(fontsize = 20)
    # if save:
        # fig.savefig('polar1mesh.png',dpi=300)

    
    py.show()

    # inn.save_weights('./network/polar1.tf')
    return None

def class1(save=False):
    input_dim = 2
    inn = INN(32,input_dim,-1.0,1.0)
    inn.load_weights("./network/class1.tf")

    inn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.99, beta_2=0.999, amsgrad=False),loss='mse',metrics=['accuracy'])

     # loc1= np.asarray([3.0,0.0])
    # loc2 = np.zeros(6)
    # low = np.ones(8) *0.04
    # high = np.ones(8) * 0.04
    # s = tfd.Sample(tfd.Normal(loc=loc1, scale=1.0))
    # s = tfd.Sample(tfd.Uniform(low=[0.1,-5.0], high=[5.0,5.0]))
    # s2 = tfd.Sample(tfd.Uniform(low=low, high=high))
    # s2 = tfd.Sample(tfd.Normal(loc=loc2, scale=0.05))

    for i in range(0):
        
        ga = gaussian.mixtureNumb(2)
        np.random.shuffle(ga)
        
        x_t = np.delete(ga,2,axis=1)
        
        _,_,y = ga.T
        y_t = np.zeros((len(y),2))
        for i in range(len(y)):
            y_t[i][int(y[i])] = 1.0
            
        # x,y = s.sample(10000).numpy().T

        # x_t = np.asarray([x,y]).T
        # r,phi = polar(x,y)
        # y_t = np.asarray([r,phi]).T
        # print(y_t)
        inn.fit(x_t,y_t,epochs=2,batch_size=32,verbose=2)



    # x,y=np.ones(10),np.zeros(10)

    # xmesh = np.linspace(-1.0,1.0,10) *0.05 + x
    # ymesh = np.linspace(-1.0,1.0,10) *0.05 + y

    # xmesh = np.repeat(xmesh,10)
    # ymesh = np.tile(ymesh,10)
    # mesh1 = np.asarray([xmesh,ymesh]).T
    # mesh2 = np.asarray([ymesh,xmesh]).T


    # xinv,yinv = inn.inv(mesh1).numpy().T
    ## py.scatter(xmesh,ymesh,c='green',alpha=0.5)
    # py.scatter(xinv,yinv,s=5.)

    # xinv,yinv = inn.inv(mesh2).numpy().T
    ##py.scatter(ymesh,xmesh,c='green',alpha=0.5)
    # py.scatter(xinv,yinv,s=5.)

    # xmesh = np.linspace(-0.3,1.3,50)
    # ymesh = np.linspace(-0.3,1.3,50) 

    # xmesh = np.repeat(xmesh,50)
    # ymesh = np.tile(ymesh,50)
    # mesh = np.asarray([xmesh,ymesh]).T
    # xinv,yinv = inn.inv(mesh).numpy().T
    # py.scatter(xinv,yinv,s=0.8)
    #-----------------------------------

    xmesh = np.linspace(-2.0,2.0,30)
    ymesh = np.linspace(-2.0,2.0,30)

    xmesh = np.repeat(xmesh,30)
    ymesh = np.tile(ymesh,30)
    mesh = np.asarray([xmesh,ymesh]).T


    # out = inn.predict(mesh)
    # outi = np.argmax(out,axis=1)
    # py.scatter(xmesh,ymesh,c=outi)

    # ga = gaussian.mixtureNumb(2)
    # x,y,_ = ga.T
    # py.scatter(x,y,c='red')

    # py.show()


    ga = gaussian.mixtureNumb(2)
    np.random.shuffle(ga)
    x_t = np.delete(ga,2,axis=1)
    xin,yin,y = ga.T
    y_t = np.zeros((len(y),2))
    for i in range(len(y)):
        y_t[i][int(y[i])] = 1.0

    xp,yp = inn.predict(x_t).T
    outinv = inn.inv(y_t).numpy()
    outinv1,outinv2 = outinv.T
    outmesh1,outmesh2 = inn.predict(mesh).T
    # print(np.mean(np.abs(y_t - out)))
    # xinv, yinv = inn.inv(y_t).numpy().T

    fig=py.figure(figsize=(11,11))
    fig.subplots_adjust(left=0.13,right = 0.95,bottom=0.1,top=0.95)
    
    ax = fig.add_subplot(111)
    ax.set_xlabel("X",fontsize=25)
    ax.set_ylabel("Y",fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=24)



    ax.scatter(xmesh,ymesh,c='green',s=80.0,label='grid input')
    ax.scatter(outmesh1,outmesh2,c='blue',s=30.0,label='grid output')
    ax.scatter(xp,yp,c='red',label='outut of training data',s=20.0)
    ax.set_xlabel("X",fontsize=25)
    ax.set_ylabel("Y",fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=25)
    if save:
        fig.savefig('class1mesh.png',dpi=300)
    py.show()
    
    # for class1_inverse.png---------------------------------------------
    # fig=py.figure(figsize=(11,3))
    # fig.subplots_adjust(left=0.2,right = 0.95,bottom=0.4,top=0.95)
    
    # ax = fig.add_subplot(111)
    # ax.set_xlabel("X",fontsize=25)
    # ax.set_ylabel("Y",fontsize=25)
    # ax.tick_params(axis='both', which='major', labelsize=24)


    # ax.scatter(xin,yin,c=y,label='input',marker='x',s=80.0)
    py.scatter(xmesh,ymesh,c='green',label='grid input')
    py.scatter(outmesh1,outmesh2,c='blue',s=3.0,label='grid output')
    # ax.scatter(outinv1,outinv2,c='red',label='inverse',s=85)
    # ax.set_xlabel("X",fontsize=25)
    # ax.set_ylabel("Y",fontsize=25)
    # ax.tick_params(axis='both', which='major', labelsize=24)
    # ax.legend(fontsize=25)
    # if save:
        # fig.savefig('class1.png',dpi=300)
    # py.show()


def class2():

    def geninput():
        ga = gaussian.mixtureNumb(4)
        np.random.shuffle(ga)
        x_t = np.delete(ga,2,axis=1)
        
        _,_,y = ga.T
        y_t = np.zeros((len(y),2))
        for i in range(len(y)):
            temp = y[i]
            for j in range(1,-1,-1):
                if temp > 0.0 and np.log2(temp) >= j :
                    y_t[i][j] = 1.0
                    temp -= 2**j
        
        return x_t, y_t
    
    input_dim = 2
    inn = INN(128,input_dim,-1.0,1.0)
    # inn.load_weights("./network/class2.tf")

    inn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.99, beta_2=0.999, amsgrad=False),loss='mse',metrics=['accuracy'])

    for i in range(1000):
        x_t, y_t = geninput()
        inn.fit(x_t,y_t,epochs=2,batch_size=32,verbose=2)
    
    inn.save_weights('./network/class2.tf')
    
    x_t, y_t = geninput()
    outx,outy = inn.inv(y_t).numpy().T
    # outx,outy = inn.predict(x_t).T
    x,y = x_t.T
    py.scatter(x,y,c='red')
    py.scatter(outx,outy,c='green')
    py.show()
    
class1(save=True)


 # loc1= np.asarray([3.0,0.0])
# loc2 = np.zeros(6)
# low = np.ones(8) *0.04
# high = np.ones(8) * 0.04
# s = tfd.Sample(tfd.Normal(loc=loc1, scale=1.0))
# s = tfd.Sample(tfd.Uniform(low=[0.1,-5.0], high=[5.0,5.0]))
# s2 = tfd.Sample(tfd.Uniform(low=low, high=high))
# s2 = tfd.Sample(tfd.Normal(loc=loc2, scale=0.05))


# x,y=np.ones(10),np.zeros(10)

# xmesh = np.linspace(-1.0,1.0,10) *0.05 + x
# ymesh = np.linspace(-1.0,1.0,10) *0.05 + y

# xmesh = np.repeat(xmesh,10)
# ymesh = np.tile(ymesh,10)
# mesh1 = np.asarray([xmesh,ymesh]).T
# mesh2 = np.asarray([ymesh,xmesh]).T


# xinv,yinv = inn.inv(mesh1).numpy().T
## py.scatter(xmesh,ymesh,c='green',alpha=0.5)
# py.scatter(xinv,yinv,s=5.)

# xinv,yinv = inn.inv(mesh2).numpy().T
##py.scatter(ymesh,xmesh,c='green',alpha=0.5)
# py.scatter(xinv,yinv,s=5.)

# xmesh = np.linspace(-0.3,1.3,50)
# ymesh = np.linspace(-0.3,1.3,50) 

# xmesh = np.repeat(xmesh,50)
# ymesh = np.tile(ymesh,50)
# mesh = np.asarray([xmesh,ymesh]).T
# xinv,yinv = inn.inv(mesh).numpy().T
# py.scatter(xinv,yinv,s=0.8)
#-----------------------------------

# xmesh = np.linspace(-2.0,2.0,30)
# ymesh = np.linspace(-2.0,2.0,30)

# xmesh = np.repeat(xmesh,30)
# ymesh = np.tile(ymesh,30)
# mesh = np.asarray([xmesh,ymesh]).T


# out = inn.predict(mesh)
# outi = np.argmax(out,axis=1)
# py.scatter(xmesh,ymesh,c=outi)

# ga = gaussian.mixtureNumb(2)
# x,y,_ = ga.T
# py.scatter(x,y,c='red')

# py.show()


# ga = gaussian.mixtureNumb(2)
# np.random.shuffle(ga)
# x_t = np.delete(ga,2,axis=1)
# xin,yin,y = ga.T
# y_t = np.zeros((len(y),2))
# for i in range(len(y)):
    # y_t[i][int(y[i])] = 1.0

# out = inn.predict(x_t)
# outinv = inn.inv(y_t).numpy()
# outinv1,outinv2 = outinv.T
# outmesh1,outmesh2 = inn.predict(mesh).T
# print(np.mean(np.abs(y_t - out)))
# xinv, yinv = inn.inv(y_t).numpy().T

# py.scatter(xin,yin,c=y,label='input',marker='x')
# py.scatter(xmesh,ymesh,c='green',label='grid input')
# py.scatter(outmesh1,outmesh2,c='blue',s=3.0,label='grid output')
# py.scatter(outinv1,outinv2,c='red',label='inverse')
# py.xlabel('x')
# py.ylabel('y')
# py.legend()
# py.show()


# out = np.argmax(out,axis=1)

# for i in range(len(out)):
    # if int(y[i]) == out[i]:
        # print('True')


# py.plot(r*np.cos(phi),r*np.sin(phi),'bo',label ='transformed')
# py.plot(xr,yr,'ro',label='inverse transform')
# py.plot(xmesh,ymesh,'go',label='original',alpha=0.5)
# py.xlabel('x')
# py.ylabel('y')
# py.legend()
# py.show()


