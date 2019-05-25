
# coding: utf-8

# In[524]:


__author__ = 'Robert Dzudzar <rdzudzar@swin.edu.au>'
__version__ = '20190511' # yyyymmdd; version datestamp of this notebook
#__datasets__ = ['des_dr1']
__keywords__ = ['Neutral Hydrogen', 'Galaxies','bokeh','Spectra']


# # Optical counterparts for the HIPASS survey
# *Robert Dzudzar*

# ### Table of contents
# * [Goals & notebook summary](#goals)
# * [Disclaimer & Attribution](#attribution)
# * [Imports & setup](#import)
# * [Authentication](#auth)
# * [First chapter](#chapter1)
# * [Resources and references](#resources)

# <a class="anchor" id="goals"></a>
# # Goals
# This notebook is for interactive exploration of the multiwavelength data, in particular: combining radio data (measured properties and HI emission line spectra) from the HI Parks All Sky Survey with the optical images. 

# # Summary
# We utilize data from the HI Parks All Sky Survey (HIPASS) presented in ...CITE... and publicly awailable at ...CITE... HIPASS data is presented in form of numerical properties of the galaxies and their HI emission line spectra. We combine HIPASS dataset with the optical galaxy images in an interactive way, therefore, explore dataset with visualization approach. 

# <a class="anchor" id="attribution"></a>
# # Disclaimer & attribution
# If you use this notebook for your published science, please acknowledge the following:
# 
# * Data Lab concept paper: Fitzpatrick et al., "The NOAO Data Laboratory: a conceptual overview", SPIE, 9149, 2014, http://dx.doi.org/10.1117/12.2057445
# 
# * Data Lab disclaimer: http://datalab.noao.edu/disclaimers.php

# <a class="anchor" id="import"></a>
# # Imports and setup

# In[42]:


# std lib
from getpass import getpass

# 3rd party
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import binned_statistic_2d
get_ipython().run_line_magic('matplotlib', 'inline')
from astropy.table import Table
from astropy import utils, io, convolution, stats
from astropy.visualization import make_lupton_rgb
import pandas as pd
import html5lib

#bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure
output_notebook()

from bokeh.plotting import figure, output_file, show, ColumnDataSource, gridplot, save
from bokeh.models import HoverTool

# Data Lab
from dl import queryClient as qc
# Data Lab
from dl import authClient as ac, queryClient as qc, storeClient as sc, helpers


# In[43]:


# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

# Either get token for anonymous user
token = ac.login('anonymous')

# ... or for authenticated user
#token = ac.login(input("Enter user name: "),getpass("Enter password: "))


# In[44]:


#FROM: DwarfGalaxyDESDR1_20171101.ipynb

# a little function to download the deepest stacked images
# adapted from R. Nikutta
def download_deepest_image(ra,dec,fov=0.1,band='g'):
    imgTable = svc.search((ra,dec), (fov/np.cos(dec*np.pi/180), fov), verbosity=2).to_table()
    print("The full image list contains", len(imgTable), "entries")
    
    sel0 = imgTable['obs_bandpass'].astype(str)==band
    sel = sel0 & ((imgTable['proctype'].astype(str)=='Stack') & (imgTable['prodtype'].astype(str)=='image')) # basic selection
    Table = imgTable[sel] # select
    if (len(Table)>0):
        row = Table[np.argmax(Table['exptime'].data.data.astype('float'))] # pick image with longest exposure time
        url = row['access_url'].decode() # get the download URL
        print ('downloading deepest stacked image...')
        image = io.fits.getdata(utils.data.download_file(url,cache=True,show_progress=False,timeout=120))

    else:
        print ('No image available.')
        image=None
        
    return image

# multi panel image plotter
def plot_images(images,geo=None,panelsize=4,bands=list('gri'),cmap=matplotlib.cm.gray_r):
    n = len(images)
    if geo is None: geo = (n,1)
        
    fig = plt.figure(figsize=(geo[0]*panelsize,geo[1]*panelsize))
    for j,img in enumerate(images):
        ax = fig.add_subplot(geo[1],geo[0],j+1)
        if img is not None:
            print(img.min(),img.max())
            ax.imshow(img,origin='lower',interpolation='none',cmap=cmap,norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=img.max()))
            ax.set_title('%s band' % bands[j])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)


# In[65]:


print(df['_RAJ2000'][40])
print(df['_DEJ2000'][40])


# In[63]:


#SIA
from pyvo.dal import sia
DEF_ACCESS_URL = "http://datalab.noao.edu/sia/des_dr1"
svc = sia.SIAService(DEF_ACCESS_URL)


band = 'g'
#rac=ra[7]
#decc=dec[7]

rac = df['_RAJ2000'][40]
decc =  df['_DEJ2000'][40]

gimage = download_deepest_image(rac, decc, fov=0.25, band=band) # FOV in deg
band = 'r'
rimage = download_deepest_image(rac, decc, fov=0.25, band=band) # FOV in deg
band = 'i'
iimage = download_deepest_image(rac, decc, fov=0.25, band=band) # FOV in deg
images=[gimage,rimage,iimage]


# In[64]:


img = make_lupton_rgb(iimage, rimage, gimage, stretch=50)
plot_images(images)
fig = plt.figure(figsize=[12,12])
ax = fig.add_subplot(1,1,1)

ax.imshow(img,origin='lower')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)


# In[529]:


# IMAGES
#http://aladin.u-strasbg.fr/AladinLite/?target=HIPASS%20J0002-03&fov=0.1&survey=P%2fDSS2%2fcolor
#import ipyaladin.aladin_widget as aypal
#https://github.com/cds-astro


# # Scrape HIPASS data and make spectrum

# In[46]:


# Load galaxy properties from HIPASS data (https://ui.adsabs.harvard.edu/abs/2004MNRAS.350.1195M/abstract)
from astropy.table import Table
HIPASS_data = Table.read('HIPASS_catalog.fit')
df = HIPASS_data.to_pandas()


# In[47]:


df.columns


# In[645]:


#print(df['_RAJ2000'])


# In[684]:


#fig = plt.figure(figsize=(10,10)) 
#ax = fig.add_subplot(111)
#im = ax.scatter(df['RVmom'], df['Sint'], marker='o', color='#9ebcda')

#ax.set_xlabel(r'Velocity $\mathrm{[km s^{-1}]}$')
#ax.xaxis.label.set_fontsize(12)
#ax.set_ylabel(r'Integrated Flux $\mathrm{[Jy km s^{-1}]}$')
#x.yaxis.label.set_fontsize(12)


# In[687]:


# Plot HIPASS survey

fig = plt.figure(figsize=(10,10)) 
ax = fig.add_subplot(111, projection="mollweide")
im = ax.scatter(np.radians(df['_RAJ2000']-180), np.radians(df['_DEJ2000']), c=df['RVmom'], cmap='viridis', s=20)

cb = plt.colorbar(im, orientation = 'horizontal', shrink = 0.8)
cb.set_label('RVmom') #Flux-weighted mean velocity of profile clipped at RVlo and RVhi 


ax.set_xlabel(r'$\mathrm{RA[\degree]}$')
ax.xaxis.label.set_fontsize(12)
ax.set_ylabel(r'$\mathrm{Dec[\degree]}$')
ax.yaxis.label.set_fontsize(12)

# Invalid value encountered probably because of X limits are +/- Pi which are both singularities on the Mollweide projection.


# In[688]:


df = df[0:6]
df


# In[48]:


import requests
import pandas as pd
#mport json
#rom pandas.io.json import json_normalize
from bs4 import BeautifulSoup
import tqdm


# In[691]:


# Edit url for x-th galaxy 

# Add strings to url from the imported HIPASS dataframe

all_s = [] # List of url-s
for galaxy in range(len(df)):
    
    # Cube string can be example 9(99) from table, however, for url request needs to be written as 009(099), thus adding 00(0)
    if len(str(df['cube'][galaxy]))==1:
        cube = ('00'+ str(df['cube'][galaxy]))
    elif len(str(df['cube'][galaxy]))==2:
        cube = ('0'+ str(df['cube'][galaxy]))
    else:
        cube = (str(df['cube'][galaxy]))
    
    s = ('http://www.atnf.csiro.au/cgi-bin/multi/release/download.cgi?cubename=/var/www/vhosts/www.atnf.csiro.au/'+
     'htdocs/research/multibeam/release/MULTI_3_HIDE/PUBLIC/H'+
     '{0}_abcde_luther.FELO.imbin.vrd&hann=1&coord={1}%3A{2}%3A{3}%2C{4}%3A{5}%3A{6}&xrange=-1281%2C12741&xaxis=optical&datasource=hipass&type=ascii'.format( 
         str(cube),  
         str(df['RAJ2000'][galaxy])[2:4], str(df['RAJ2000'][galaxy])[5:7], 
         str(df['RAJ2000'][galaxy])[8:10], str(df['DEJ2000'][galaxy])[2:5], str(df['DEJ2000'][galaxy])[6:8], str(df['DEJ2000'][galaxy])[9:11] ) )
    all_s.append(s)
    print(s)


# In[612]:


#http://www.atnf.csiro.au/cgi-bin/multi/release/download.cgi?cubename=/var/www/vhosts/www.atnf.csiro.au/htdocs/research/multibeam/release/MULTI_3_HIDE/PUBLIC/H006_abcde_luther.FELO.imbin.vrd&hann=1&coord=15%3A48%3A13.1%2C-78%3A09%3A16&xrange=-1281%2C12741&xaxis=optical&datasource=hipass&type=ascii


# In[60]:


# SKYMAPPER

image = 'http://api.skymapper.nci.org.au/aus/siap/dr2/get_image?IMAGE=20140425124821-10&SIZE=0.0833&POS=189.99763,-11.62305&FORMAT=png'
import IPython
url = image
IPython.display.Image(url, width = 250)


# In[618]:


# Go to url and get the spectrum data
#res = requests.get("http://www.atnf.csiro.au/cgi-bin/multi/release/download.cgi?cubename=/var/www/vhosts/www.atnf.csiro.au/htdocs/research/multibeam/release/MULTI_3_HIDE/PUBLIC/H006_abcde_luther.FELO.imbin.vrd&hann=1&coord=15%3A48%3A13.1%2C-78%3A09%3A16&xrange=2100%2C3100&xaxis=optical&datasource=hipass&type=ascii")
#res = requests.get("http://www.atnf.csiro.au/cgi-bin/multi/release/download.cgi?cubename=/var/www/vhosts/www.atnf.csiro.au/htdocs/research/multibeam/release/MULTI_3_HIDE/PUBLIC/H006_abcde_luther.FELO.imbin.vrd&hann=1&coord=15%3A48%3A13.1%2C-78%3A09%3A16&xrange=-1281%2C12741&xaxis=optical&datasource=hipass&type=ascii")


Intensity = []
Velocity = []
Channel = []
count = -1
for each_galaxy in all_s:    
    count += 1
    res = requests.get(each_galaxy)
    soup = BeautifulSoup(res.content,'lxml')
    
    # Take a part of the url page where the spectral information is held
    start_df = str(soup)[1510:]  #starting number information
    for_table = start_df[:50176] #end reading before (</p></body></html>)
    
    # Split data into rows with separator '\n' 
    a = for_table.rstrip().split('\n')
    
    # Go line by line and extract string which are actually numbers (3 columns)
    Chan = []
    Vel = []
    Int = []
    for i in a:
        Chan.append(i[1:12])
        Vel.append(i[17:33])
        Int.append(i[36:49])
        
    # Convert string into floats
    I = [float(i) for i in Int]
    C = [float(i) for i in Chan]
    V = [float(i) for i in Vel]
    Intensity.append(I)
    Velocity.append(V)
    Channel.append(C)
    
    #fig = plt.figure(figsize=(8,7))                                                               
    #ax = fig.add_subplot(1,1,1)
    #plt.plot(V, I, 'k', linewidth = 1) # Plot spectrum
    #plt.xlim(df['RV1'][count]-500, df['RV2'][count]+500) # Read in from table
    
    #plt.axvline(df['RVsp'][count]) # Add peak line
    #ax.axvspan(df['RV1'][count], df['RV2'][count], ymin=0, ymax=1, alpha=0.5, color='lightgrey') # Shade spectra region
    
    #plt.ylim(-0.05, 0.1) #read in from table
    #plt.ylabel('Flux density [Jy beam$^{-1}$]', fontsize = 15)
    #plt.xlabel('Optical Velocity [km s$^{-1}$]', fontsize = 15)
    
    #ax.get_yaxis().set_tick_params(which = 'both', direction='in', right = True, size = 8)
    #ax.get_xaxis().set_tick_params(which = 'both', direction='in', top = True, size = 8)
    #plt.show()


# In[622]:


# Plot the spectrum
for i in range(len(Velocity)):
    
    fig = plt.figure(figsize=(8,7))                                                               
    ax = fig.add_subplot(1,1,1)
    plt.plot(Velocity[i], Intensity[i], 'k', linewidth = 1) # Plot spectrum
    plt.xlim(df['RV1'][i]-500, df['RV2'][i]+500) # Read in from table
    
    #plt.axvline(df['RVsp'][count]) # Add peak line
    ax.axvspan(df['RV1'][i], df['RV2'][i], ymin=0, ymax=1, alpha=0.5, color='lightgrey') # Shade spectra region
    
    plt.ylim(-0.05, 0.1) #read in from table
    plt.ylabel('Flux density [Jy beam$^{-1}$]', fontsize = 15)
    plt.xlabel('Optical Velocity [km s$^{-1}$]', fontsize = 15)
    
    ax.get_yaxis().set_tick_params(which = 'both', direction='in', right = True, size = 8)
    ax.get_xaxis().set_tick_params(which = 'both', direction='in', top = True, size = 8)
    plt.show()


# In[19]:



from astroquery.vizier import Vizier
