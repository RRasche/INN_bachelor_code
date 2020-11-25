import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as py


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def single_hidden_layer(save=False):
    fig = py.figure(figsize=(14,7))
    ax = fig.add_subplot(111)
    ax.set_xlabel("X",fontsize=20)
    ax.set_ylabel("Y",fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.get_major_formatter()._usetex = False 
    ax.yaxis.get_major_formatter()._usetex = False
    x = np.linspace(-2.0,2.0,1000)
    w = [5.0,10.0,50.0,1000.0]
    for weight in w:
        b = 0.0
        z = x*weight + b
        ax.plot(x,sigmoid(z),linewidth=5.0,label='$w = $ ' + str(weight))
    ax.legend(fontsize=20)
    if(save):
        fig.savefig('single_hidden_layer.png',dpi=300)
    
    py.show()
    
def two_hidden_layer(save=False):
    fig = py.figure(figsize=(14,7))
    ax = fig.add_subplot(111)
    ax.set_xlabel("X",fontsize=20)
    ax.set_ylabel("Y",fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.get_major_formatter()._usetex = False 
    ax.yaxis.get_major_formatter()._usetex = False
    x = np.linspace(-2.0,2.0,1000)
    out = np.zeros_like(x)
    w = [1000.0,1000.0]
    b = [-1.0,1.0]
    a=[1.0,-1.0]
    for weight,bias,alpha in zip(w,b,a):
        bias = -bias*weight
        z = x*weight + bias
        out += alpha*sigmoid(z)
        
    ax.plot(x,out,linewidth=5.0)
    # ax.legend(fontsize=20)
    if(save):
        fig.savefig('single_hidden_layer.png',dpi=300)
    
    py.show()
    
def approx_sin(size=50,save=False):
    fig = py.figure(figsize=(14,7))
    ax = fig.add_subplot(111)
    ax.set_xlabel("X",fontsize=20)
    ax.set_ylabel("Y",fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.get_major_formatter()._usetex = False 
    ax.yaxis.get_major_formatter()._usetex = False
    
    x = np.linspace(-10.0,10.0,1000)
    out = np.zeros_like(x)
    w = np.ones(size) *1000.0
    b = np.linspace(-10.0,10.0,size)
    a = np.zeros_like(b)
    
    
    curr_val = 0.0
    for i in range(len(a)):
        curr_sin = np.sin(b[i])
        update = curr_sin - curr_val
        curr_val += update
        a[i] = update
    
    for weight,bias,alpha in zip(w,b,a):
        bias = -bias*weight
        z = x*weight + bias
        out += alpha*sigmoid(z)
        
    ax.plot(x,out,label="Network",linewidth=5.0)
    ax.plot(x,np.sin(x),label='$sin(x)$',linewidth=5.0)
    ax.legend(fontsize=20)
    if(save):
        fig.savefig('approx_sin.png',dpi=300)
    
    py.show()
    
approx_sin(save = True)