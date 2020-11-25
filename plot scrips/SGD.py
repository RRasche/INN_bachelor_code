import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as py


def para(x,y):
    return np.power(x,3) + np.power(y,3)
    
    
size = 15
x,y = np.linspace(0,3,size),np.linspace(0,3,size)
x = np.reshape(np.tile(x,(size)),(size,size))
y=np.tile(np.reshape(y,(size,1)),(1,size))
z = para(x,y)

points = np.asarray([[2.9,2.9, para(2.9,2.9)+1],[2.6,2.8, para(2.6,2.8)+1],[2.7,2.5, para(2.7,2.5)+1],[2.4,2.4, para(2.4,2.4) + 1],[2.3,2.1, para(2.3,2.1) + 1],[1.9,2.2, para(1.9,2.2) + 1]])
a = 0.2
points2 = np.asarray([[2.9,2.9, para(2.9,2.9)+1],[2.9-a,2.9-a, para(2.9-a,2.9-a)+1]])

fig = py.figure()
ax = fig.gca(projection='3d')
fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
ax.view_init(20, 250)
ax.set_xlabel("X",fontsize=23)
ax.set_ylabel("Y",fontsize=23)
ax.set_zlabel("Z",fontsize=23)
ax.tick_params(axis='both', which='major', labelsize=23)
ax.plot_surface(X = x,Y = y,Z = z,cmap='gist_gray',antialiased=True,zorder=1)
ax.plot(points[:,0],points[:,1],points[:,2],'-', linewidth=3,c='black',antialiased=True,zorder=3,label='stochastic gradient descent')
ax.plot(points2[:,0],points2[:,1],points2[:,2],'-', linewidth=3,c='red',antialiased=True,zorder=3,label='gradient descent')
ax.legend(loc='upper left', fontsize=18)
fig.savefig('SGD.png',dpi=300)
py.show()