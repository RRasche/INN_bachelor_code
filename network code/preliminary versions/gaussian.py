import numpy as np
import matplotlib.pyplot as py

def mixture(numb,size = 1600):
    #Input:
    #   numb <int> determines the amount of modes/distributions the Gaussian mixture model incorporates.
    #   size <int> determines the amount of samples per distribution
    #Output:
    #   ga <2D numpy array> 
    #   ga[0,:] <float numpy array> contains the x coordinates
    #   ga[1,:] <float numpy array> contains the y coordinates
    #   ga[2,:] <float numpy array> contains the labels, which show to which distribution the (x,y) tupel belongs
    #
    #This function creates a Gaussian mixture model and returns x and y coordinates as well as labels.
    ga = np.zeros((size*numb,3))
    for i in range(numb):
        ga[i*size:(i+1)*size] = np.concatenate((np.random.normal(2.42*np.cos(i/numb * 2.0*np.pi),0.2,(size,1)),np.random.normal(2.42*np.sin(i/numb * 2.0*np.pi),0.2,(size,1)),np.ones((size,1)) * i ),axis=1)
    return ga
