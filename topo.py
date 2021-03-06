#!/usr/bin/env python
from netCDF4 import Dataset
import scipy.interpolate
import numpy as np
import tplot.cm
from tplot.xsection import cmap_to_rgba
import mayavi.mlab 

def draw_topography(basemap,zscale=1,cmap=tplot.cm.etopo1,**kwargs):
  url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo1.nc'
  etopodata = Dataset(url)

  lonmin = basemap.lonmin
  lonmax = basemap.lonmax
  latmin = basemap.latmin
  latmax = basemap.latmax
  lons = etopodata.variables['x'][:]
  lats = etopodata.variables['y'][:]

  xidx = (lons > lonmin) & (lons < lonmax)
  yidx = (lats > latmin) & (lats < latmax)
  lons = lons[xidx]
  lats = lats[yidx]

  topoin = etopodata.variables['rose'][yidx,xidx]
  topoin = np.ma.masked_array(topoin,np.isnan(topoin))
  topoin.data[topoin.mask] = 0.0

  #data = topoin.data                                                                                              
  itp = scipy.interpolate.interp2d(lons,lats,topoin,kind='cubic')

  lonitp = np.linspace(min(lons),max(lons),3*len(lons))
  latitp = np.linspace(min(lats),max(lats),3*len(lats))

  dataitp = itp(lonitp,latitp)
  longrid,latgrid = np.meshgrid(lonitp,latitp)
  xgrid,ygrid = basemap(longrid,latgrid)
  vmin = -8210.0*zscale
  vmax = 7000.0*zscale
  m = mayavi.mlab.mesh(xgrid,ygrid,zscale*dataitp,vmin=vmin,vmax=vmax,
                       **kwargs)
  rgba = cmap_to_rgba(cmap)
  m.module_manager.scalar_lut_manager.lut.table = rgba
  mayavi.mlab.draw()



