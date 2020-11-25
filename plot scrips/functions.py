import numpy as np
import matplotlib.pyplot as py

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
    
    
x = np.linspace(-10,10,1000)    
fig = py.figure(figsize=(15,10))
ax = fig.add_subplot(111)
ax.set_xlabel("X",fontsize=30)
ax.set_ylabel("Y",fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.plot(x,sigmoid(x),linewidth=5.0)
fig.savefig('sigmoid.png',dpi=300)