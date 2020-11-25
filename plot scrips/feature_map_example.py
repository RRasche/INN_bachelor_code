import numpy as np
import matplotlib.pyplot as py
from mpl_toolkits.mplot3d import Axes3D

def mixture():
    numb = 1600
    ga = np.zeros((numb*3,3))
    for i in range(3):
        ga[i*numb:(i+1)*numb] = np.concatenate((np.random.normal(2.0*(i-1),0.2,(numb,1)),np.random.normal(2.0*(i-1),0.2,(numb,1)),np.ones((numb,1)) * np.abs(i-1) ),axis=1)
    return ga

def plotEmb(x,y,z,save=False):

    z_emb =np.power(x,2) + np.power(y,2)
    
    fig = py.figure(figsize=(11,8))
    ax = fig.add_subplot(111,projection='3d')
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    ax.view_init(10, 125)
    ax.set_xlabel("X",fontsize=23)
    ax.set_ylabel("Y",fontsize=23)
    ax.set_zlabel("Z",fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=23)
    
    ax.scatter(x,y,z_emb,c=z,cmap='brg')
    
    if(save):
        fig.savefig('feature_map_3D.png',dpi=300)
    py.show()

def plotData(x,y,z,save=False):
    fig = py.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel("X",fontsize=25)
    ax.set_ylabel("Y",fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25)
    py.scatter(x,y,c=z,cmap='brg')
    if(save):
        fig.savefig('feature_map_2D.png',dpi=300)
    py.show()
    
x,y,z = mixture().T
# plotData(x,y,z,save=True)
plotEmb(x,y,z,save=True)