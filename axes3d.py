#!/usr/bin/env python
from misc import rotation3D
from mpl_toolkits.mplot3d import Axes3D as _Axes3D
from matplotlib import cm
import numpy as np
import cm as tcm
import matplotlib.pyplot as plt

class Axes3D(_Axes3D):
  def __init__(self,*args,**kwargs):
    if (len(args) > 1) | kwargs.has_key('rect'):
      _Axes3D.__init__(self,*args,**kwargs)

    else:
      rect = (0.1,0.1,0.8,0.8)
      _Axes3D.__init__(self,*args,rect=rect,**kwargs)
    
  def cross_section(self,func,anchor,
                    zrot,yrot,xrot,
                    length,width,
                    Nl=20,Nw=20,
                    cmap=tcm.slip,
                    clim=None,
                    func_args=None,
                    func_kwargs=None):
    if func_args is None:
      func_args = ()

    if func_kwargs is None:
      func_kwargs = {}

    x = np.linspace(0,length,Nl)
    y = np.linspace(0,-width,Nw)
    x,y = np.meshgrid(x,y)
    z = 0.0*x
    R = rotation3D(zrot,yrot,xrot)
    p = np.concatenate((x[None,:,:],
                        y[None,:,:],
                        z[None,:,:]),
                        axis=0)

    p = np.einsum('ij,jkl->kli',R,p)
    p += anchor
    #p = np.einsum('ijk->kij',p)
    p = np.reshape(p,(Nl*Nw,3))   
    #x = p[:,:,0]
    #y = p[:,:,1]
    #z = p[:,:,2]
    c = func(p,*func_args,**func_kwargs)
    c = np.reshape(c,(Nl,Nw))
    p = np.reshape(p,(Nl,Nw,3))
    if clim is None:
      clim = (np.min(c),np.max(c))

    cnorm = (c - clim[0])/(clim[1] -clim[0])
    s = self.plot_surface(p[:,:,0],
                          p[:,:,1],
                          p[:,:,2],
                          shade=False,
                          facecolors=cmap(cnorm),
                          rstride=1,cstride=1)
     
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(c)

    idx1 = np.array([[0,-1],[0,-1]])
    idx2 = np.array([[0,0],[-1,-1]])
    w = self.plot_wireframe(p[:,:,0][idx1,idx2],
                            p[:,:,1][idx1,idx2],
                            p[:,:,2][idx1,idx2],
                            color='k')
    return m
    
def f(points):
  return np.sqrt(np.sum(points**2,1))

fig = plt.figure()
ax = Axes3D(fig)
xrot = np.pi/2.0001
yrot = 0.0
zrot = 0.0
length = 2.0
width = 2.0
anchor = np.array([-1.0,0.0,1.0])



q = ax.cross_section(f,anchor,
                 zrot,yrot,xrot,
                     length,width,cmap=tcm.slip_r,Nl=50,Nw=50)
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_zlim(-3,3)
plt.colorbar(q)
plt.show()
  
