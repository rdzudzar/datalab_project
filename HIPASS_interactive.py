
# coding: utf-8

# In[1]:


__author__ = 'Robert Dzudzar <robertdzudzar@gmail.com>, <rdzudzar@swin.edu.au>'
__version__ = '20190511' # yyyymmdd; version datestamp of this notebook
#__datasets__ = ['des_dr1']
__keywords__ = ['Neutral Hydrogen', 'Galaxies','bokeh','Spectra']


# # Interactively examining the HI Parkes All Sky Survey (HIPASS)
# *Robert Dzudzar*

# ### Table of contents
# * [Goals & notebook summary](#goals)
# * [Disclaimer & Attribution](#attribution)
# * [Imports & setup](#import)
# * [Authentication](#auth)
# * [Import HIPASS data](#chapter1)
#     * [Plot the Sky coverage of the HIPASS survey](#chapter1.1)
#     * [Make a smaller dataset of HIPASS data](#chapter1.2)
#     * [Scraping url-s where the data of the HIPASS spectra is storred](#chapter1.3)
#     * [Creating list of HIPASS sources](#chapter1.4)
#     * [Extracting spectral information from HIPASS database](#chapter1.5)
#     * [Plotting the HI spectra for each source](#chapter1.6)
# * [Query optical counterparts of the HIPASS sources](#chapter2)
#     * [Create a list of coordinates](#chapter2.1)
#     * [Download and save images from SkyView](#chapter2.2)
# * [Setting up Bokeh for interactive examination](#chapter3)
#     * [Extracting/Sorting images and spectra which we have](#chapter3.1)
#     * [Approximate Distance and HI mass](#chapter3.2)
# * [Interactive visualization with Bokeh](#chapter4)
# * [Resources and references](#resources)

# <a class="anchor" id="goals"></a>
# # Goals
# This notebook is for interactive exploration of the multiwavelength data, in particular: a combination of the radio data (measured properties and HI emission line spectra) from the HI Parks All Sky Survey and the optical data. 

# # Summary
# We utilize data from the HI Parks All Sky Survey (HIPASS) presented in https://ui.adsabs.harvard.edu/?#abs/2004MNRAS.350.1195M and publicly awailable at http://www.atnf.csiro.au/research/multibeam/release/. The HIPASS data are presented in the form of numerical properties of the sources (galaxies) and their HI emission line spectra. We obtain the HI spectra from the HIPASS database and query their optical images from SkyView. The both datasets are then combined in an interactive environment, which enables numerical and visual examination of the data. 

# <a class="anchor" id="attribution"></a>
# # Disclaimer & attribution
# If you use this notebook for your published science, please acknowledge the following:
# 
# * Data Lab concept paper: Fitzpatrick et al., "The NOAO Data Laboratory: a conceptual overview", SPIE, 9149, 2014, http://dx.doi.org/10.1117/12.2057445
# 
# * Data Lab disclaimer: http://datalab.noao.edu/disclaimers.php

# <a class="anchor" id="import"></a>
# # Imports and setup

# In[2]:


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

 # bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, show, ColumnDataSource, gridplot, save
from bokeh.models import HoverTool
output_notebook()

# Data Lab
from dl import queryClient as qc
# Data Lab
from dl import authClient as ac, queryClient as qc, storeClient as sc, helpers


# In[3]:


# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

# Either get token for anonymous user
token = ac.login('anonymous')

# ... or for authenticated user
#token = ac.login(input("Enter user name: "),getpass("Enter password: "))


# In[4]:


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


# In[5]:


#SIA
from pyvo.dal import sia
DEF_ACCESS_URL = "http://datalab.noao.edu/sia/des_dr1"
svc = sia.SIAService(DEF_ACCESS_URL)


band = 'g'
#rac=ra[7]
#decc=dec[7]

rac = df['_RAJ2000'][59]
decc =  df['_DEJ2000'][59]

gimage = download_deepest_image(rac, decc, fov=0.25, band=band) # FOV in deg
band = 'r'
rimage = download_deepest_image(rac, decc, fov=0.25, band=band) # FOV in deg
band = 'i'
iimage = download_deepest_image(rac, decc, fov=0.25, band=band) # FOV in deg
images=[gimage,rimage,iimage]


# In[ ]:


img = make_lupton_rgb(iimage, rimage, gimage, stretch=50)
plot_images(images)
fig = plt.figure(figsize=[12,12])
ax = fig.add_subplot(1,1,1)

ax.imshow(img,origin='lower')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)


# <a class="anchor" id="chapter1"></a>
# # Import HIPASS data

# In[3]:


# Load galaxy properties from HIPASS data (https://ui.adsabs.harvard.edu/abs/2004MNRAS.350.1195M/abstract)
from astropy.table import Table
HIPASS_data = Table.read('HIPASS_catalog.fit')
df_hipass = HIPASS_data.to_pandas()


# In[56]:


# Display the dataframe
df_hipass
# You can see all the columns:
#df_hipass.columns


# <a class="anchor" id="chapter1.1"></a>
# ## Plot the Sky coverage of the HIPASS survey

# In[5]:


# Plot HIPASS survey

fig = plt.figure(figsize=(14,14)) 
ax = fig.add_subplot(111, projection="mollweide") # Using mollweide projection
# Converting RA and DEC from deg to radians
im = ax.scatter(np.radians(df_hipass['_RAJ2000']-180), np.radians(df_hipass['_DEJ2000']), c=df_hipass['RVmom'], cmap='viridis', s=20)
# Adding colorbar for the sources, based on their Velocity
cb = plt.colorbar(im, orientation = 'horizontal', shrink = 0.8)
# RVmom ==> Flux-weighted mean velocity of profile clipped at RVlo and RVhi (explained in the online HIPASS table)
cb.set_label(r'RVmom [km s$^{-1}$]') 


ax.set_xlabel(r'$\mathrm{RA[\degree]}$')
ax.xaxis.label.set_fontsize(12)
ax.set_ylabel(r'$\mathrm{Dec[\degree]}$')
ax.yaxis.label.set_fontsize(12)

# Invalid value encountered probably because of x limits are +/- Pi which are both singularities on the Mollweide projection.


# <a class="anchor" id="chapter1.2"></a>
# ## Make a smaller dataset of HIPASS data

# In[57]:


# For faster download of the optical images and HI spectra, it is recommended to start with smaller sample
# Pre-compiled 500 takes a long to run (~1hour) from scratch
# Small sub-sample of the HIPASS data
df = df_hipass[0:500]
df


# In[7]:


import requests
#mport json
#rom pandas.io.json import json_normalize
from bs4 import BeautifulSoup


# <a class="anchor" id="chapter1.3"></a>
# ## Scraping url-s where the data of the HIPASS spectra is storred

# In[8]:


# Edit url for each galaxy in HIPASS: for making url-s we need: RA, DEC, and a number of the cube from where data was extracted
# Needed data are provided in the HIPASS table: df['RAJ2000'], df['DEJ2000'] and df['cube']

# List of url-s
all_s = [] 
# Go through each galaxy from the dataframe
for galaxy in range(df.index[0], df.index[0]+len(df)):
    
    # Cube string can be example 9(99) from table, however, for url request they need to be written as 009(099)
    # We check the cube number length and add 00(0) if needed.
    
    if len(str(df['cube'][galaxy]))==1:
        cube = ('00'+ str(df['cube'][galaxy]))
    elif len(str(df['cube'][galaxy]))==2:
        cube = ('0'+ str(df['cube'][galaxy]))
    else:
        cube = (str(df['cube'][galaxy]))
    
    # We combine all aquired strings into the uls string which is constant and append it to `all_s`
    s = ('http://www.atnf.csiro.au/cgi-bin/multi/release/download.cgi?cubename=/var/www/vhosts/www.atnf.csiro.au/'+
     'htdocs/research/multibeam/release/MULTI_3_HIDE/PUBLIC/H'+
     '{0}_abcde_luther.FELO.imbin.vrd&hann=1&coord={1}%3A{2}%3A{3}%2C{4}%3A{5}%3A{6}&xrange=-1281%2C12741&xaxis=optical&datasource=hipass&type=ascii'.format( 
         str(cube),  
         str(df['RAJ2000'][galaxy])[2:4], str(df['RAJ2000'][galaxy])[5:7], 
         str(df['RAJ2000'][galaxy])[8:10], str(df['DEJ2000'][galaxy])[2:5], str(df['DEJ2000'][galaxy])[6:8], str(df['DEJ2000'][galaxy])[9:11] ) )
    all_s.append(s)
    
    # Print each created url, can open them and see how the data looks like 
    print(s)


# In[9]:


#http://www.atnf.csiro.au/cgi-bin/multi/release/download.cgi?cubename=/var/www/vhosts/www.atnf.csiro.au/htdocs/research/multibeam/release/MULTI_3_HIDE/PUBLIC/H006_abcde_luther.FELO.imbin.vrd&hann=1&coord=15%3A48%3A13.1%2C-78%3A09%3A16&xrange=-1281%2C12741&xaxis=optical&datasource=hipass&type=ascii


# <a class="anchor" id="chapter1.4"></a>
# ##  Creating list of HIPASS sources

# In[10]:


# Extract the HIPASS source names from the table; String manipulation is needed to strip certain characters from name 
# Also, each source has sting `HIPASS` in front of its table name, so we add that

# Creating HIPASS sources name list
HIPASS_sources = []

for galaxy_name in range(df.index[0], df.index[0]+len(df)):
    gal_name = str(df['HIPASS'][galaxy_name]).strip('b\' ')
    HIPASS_sources.append('HIPASS'+gal_name)
print(HIPASS_sources)


# <a class="anchor" id="chapter1.5"></a>
# ## Extracting spectral information from HIPASS database

# In[11]:


# We want to go to each url and extract only the spectra data
# From each url we need Intensity, Velocity and Channel information

#res = requests.get("http://www.atnf.csiro.au/cgi-bin/multi/release/download.cgi?cubename=/var/www/vhosts/www.atnf.csiro.au/htdocs/research/multibeam/release/MULTI_3_HIDE/PUBLIC/H006_abcde_luther.FELO.imbin.vrd&hann=1&coord=15%3A48%3A13.1%2C-78%3A09%3A16&xrange=2100%2C3100&xaxis=optical&datasource=hipass&type=ascii")
#res = requests.get("http://www.atnf.csiro.au/cgi-bin/multi/release/download.cgi?cubename=/var/www/vhosts/www.atnf.csiro.au/htdocs/research/multibeam/release/MULTI_3_HIDE/PUBLIC/H006_abcde_luther.FELO.imbin.vrd&hann=1&coord=15%3A48%3A13.1%2C-78%3A09%3A16&xrange=-1281%2C12741&xaxis=optical&datasource=hipass&type=ascii")

#Storring values
Intensity = []
Velocity = []
Channel = []

# Going through each url, reading it with the BeautifulSoup and manipulating to get needed data
count = -1
for each_galaxy in all_s:    
    count += 1
    res = requests.get(each_galaxy)
    soup = BeautifulSoup(res.content,'lxml')
    
    # Take a part of the url page where the spectral information is held
    # This requires skipping first 1510 characters and everything after 50176 character from the url 
    # These numbers are always the same for the HIPASS database
    
    start_df = str(soup)[1510:]  # Starting character 
    for_table = start_df[:50176] # Remove everything after this character (</p></body></html>)
    
    # Split data into rows with separator '\n' 
    a = for_table.rstrip().split('\n')
    
    # Go line by line and extract string which are actually numbers (3 columns: Channel, Velocity and Intensity)
    Chan = []
    Vel = []
    Int = []
    # We know where the required informations are storred so we are just extracting certain characters
    for i in a:
        Chan.append(i[1:12])
        Vel.append(i[17:33])
        Int.append(i[36:49])
        
    # Convert string into floats and save them for each galaxy
    I = [float(i) for i in Int]
    C = [float(i) for i in Chan]
    V = [float(i) for i in Vel]
    Intensity.append(I)
    Velocity.append(V)
    Channel.append(C)
    


# <a class="anchor" id="chapter1.6"></a>
# ## Plotting the HI spectra for each source

# In[40]:


# Plot the spectra of all sources and save files in a subdirectory - these will be used for interactive examination
# For each source that was extracted
store_indices = []
for idx, i in enumerate(range(len(Velocity))):
#for i in range(len(Velocity)):
    store_indices.append(idx)
    fig = plt.figure(figsize=(8,7))                                                               
    ax = fig.add_subplot(1,1,1)
    
    # Plotting Velocity and HI intensity
    plt.plot(Velocity[i], Intensity[i], 'k', linewidth = 1, label=str(df['HIPASS'][i]).strip('b\' ') )
    # Read the position where HI is detected (information from the table)
    # Adding by default velocity +- from the center of the detected source velocity
    # Range can be arbitrary as the velocity information for each source is in range from around -1280 to around 12726 km/s
    plt.xlim(df['RV1'][i]-500, df['RV2'][i]+500) 
    
    # plt.axvline(df['RVsp'][count]) # Add peak line
    # Adding span in which HI spectrum was integrated to get the flux values
    ax.axvspan(df['RV1'][i], df['RV2'][i], ymin=0, ymax=1, alpha=0.5, color='lightgrey') # Shade spectra region
    
    # Add limits to plot, labels, ticks and save figure
    # For limits on y-axis, use Speak

    plt.ylim(-0.05, df['Speak'][i]+0.02)
    plt.ylabel('Flux density [Jy beam$^{-1}$]', fontsize = 15)
    plt.xlabel('Optical Velocity [km s$^{-1}$]', fontsize = 15)
    
    ax.get_yaxis().set_tick_params(which = 'both', direction='in', right = True, size = 13)
    ax.get_xaxis().set_tick_params(which = 'both', direction='in', top = True, size = 13)
    
    plt.legend(loc=1, fontsize=20)
    fig.savefig('./HIPASS_spectra/{0}.png'.format(idx), overwrite=True)
    #plt.show()


# In[13]:


from astroquery.vizier import Vizier


# <a class="anchor" id="chapter2"></a>
# # Query optical counterparts of the HIPASS sources

# In[14]:


# Using SkyView to get the DSS images of the sources
#from astroquery.skyview import SkyView


# In[15]:


#list all available image data which can be obtained from SkyView
SkyView.list_surveys() 

# For DSS: 
#'Optical:DSS': ['DSS',
#                  'DSS1 Blue',
#                  'DSS1 Red',
#                  'DSS2 Red',
#                  'DSS2 Blue',
#                  'DSS2 IR'],


# In[16]:


from astropy import coordinates, units as u, wcs
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
import pylab as pl


# <a class="anchor" id="chapter2.1"></a>
# ## Create a list of coordinates

# In[17]:


# To query Sky position (images) of sources, we need central position of each detection in HIPASS so we extract them using SkyCoord

#c = []
#for each_galaxy in HIPASS_sources:
#    center = coordinates.SkyCoord.from_name(each_galaxy)
#    c.append(center)
#    #print(center)
    
c = []
for each_galaxy in df.index:
    center = SkyCoord(df['_RAJ2000'][each_galaxy], df['_DEJ2000'][each_galaxy], frame='icrs', unit="deg")
    c.append(center)
    #print(center)


# <a class="anchor" id="chapter2.2"></a>
# ## Download and save images from SkyView
# ### (This cell takes long to run for large number of sources (~1h for 500))

# In[18]:


from urllib.error import HTTPError
TIMEOUT_SECONDS = 36000
# Get image from the SkyView based on the position
# Radius of the extracted images is matched to the HIPASS primary beam (~15 arcmin)
# Attention --- Not all HIPASS sources are clearly identified, since the beam is 15 arcmin there are confused sources - thus
# possible optical counterpart will be off center in the optical image

for idx, each_galaxy in enumerate(HIPASS_sources):
#for each_galaxy in HIPASS_sources:

# Encountering  HTTPError when the position of the source is not in the image database, then it will be skipped and user will be notified   
    try:
        # Get coordinates
        center = coordinates.SkyCoord.from_name(each_galaxy)
        
        # Get image from the SkyView based on coordinates; radius is matched to HIPASS primary beam
        Survey = 'DSS'
        images = SkyView.get_images(position=center, pixels=[1000,1000], survey=Survey, radius=5*u.arcmin)
        
        image = images[0]
        
        # 'imgage' is now a fits.HDUList object; the 0th entry is the image
        mywcs = wcs.WCS(image[0].header)
        
        fig = pl.figure(figsize=(8,8))
        fig.clf() # just in case one was open before
        
        # use astropy's wcsaxes tool to create an image with RA/DEC on axis
        ax = fig.add_axes([0.15, 0.1, 0.8, 0.8], projection=mywcs)
        
        ax.set_xlabel("RA", fontsize=15)
        ax.set_ylabel("Dec", fontsize=15)
        
        # Show image
        
        ax.imshow(image[0].data, cmap='gray_r', interpolation='none', origin='lower',
                  norm=pl.matplotlib.colors.LogNorm())
        
        matplotlib.rcParams.update({'font.size': 22})
        
        #ax.get_yaxis().set_tick_params(which = 'both', direction='in', right = True, size = 8, labelsize=20)
        #ax.get_xaxis().set_tick_params(which = 'both', direction='in', top = True, size = 8)
    
        fig.savefig('./HIPASS_images/{0}.png'.format(idx), overwrite=True)
        
    except HTTPError:
        print('Image not found in the {0} filter'.format(Survey))
        continue
    
    
    #ax.axis([100, 200, 100, 200])


# <a class="anchor" id="chapter3"></a>
# # Setting up Bokeh for interactive examination

# In[58]:


from bokeh.plotting import figure, output_file, show, ColumnDataSource, gridplot, save
from bokeh.models import HoverTool, BoxSelectTool
output_notebook()


# <a class="anchor" id="chapter3.1"></a>
# ### Extracting/Sorting images and spectra which we have

# In[20]:


# Check files in the downloaded folder and play around with the strings to sort them and extract

from os import listdir

extension = '.png'
mypath = r'./HIPASS_images/'
# Get all the files from a directory with extension
files_with_extension = [ f for f in listdir(mypath) if f[(len(f) - len(extension)):len(f)].find(extension)>=0 ]
# Strip the extension from the files
raw_image_indices = [x.rstrip('.png') for x in files_with_extension]

# For all the files now we can use integer sorting so that we obtain: 01 02 03 and not 01 10 etc.
raw_image_indices = [int(x) for x in raw_image_indices]
sorted_list_of_images = sorted(raw_image_indices)
#print(sorted_list_of_images)

# For the new created list we are adding now image/spectra location and .png
# We create new arrays with the files
new_list_sorted = []
new_list_images = []
for i in sorted_list_of_images:
    New_list_s = './HIPASS_spectra/'+str(i)+'.png'
    New_list_i = './HIPASS_images/'+str(i)+'.png'
    new_list_sorted.append(New_list_s)
    new_list_images.append(New_list_i)
#print(new_list_sorted)


# <a class="anchor" id="chapter3.2"></a>
# ### Approximate Distance and HI mass

# In[21]:


# HI mass approximation only! here we assume RVmom is recessional velocity!
# Also using approximate distance as: RVmom * H0

H0 = 70 # Hubble constant
Distance = df['RVmom']/H0
HI_mass = np.log10(2.365*10e5*(Distance**2)*df['Sint'])


# <a class="anchor" id="chapter4"></a>
# # Interactive visualization with Bokeh

# In[55]:


# Add bokeh features
# We are plotting x and y data
# As desc - description - we will have name of the object
# as spectra and imgs -- we will have spectrum image and optical image for each source as we hover above plotted points
# Depending on the speed of your internet, when first time hovering on points - wait a couple of seconds for images to appear

import matplotlib as mpl
from bokeh.palettes import BuGn8, viridis
from bokeh.transform import linear_cmap
from bokeh.models import ColumnDataSource, ColorBar

source = ColumnDataSource(
        data=dict(
            x = Distance,
            y = HI_mass, 
            z = df['W20max'],
            desc = HIPASS_sources ,
            confused = df['cf'],
            Int = df['Sint'],
            spectra = new_list_sorted,
            imgs = new_list_images,))

# Adding html code to say how the images and source name will be displayed
hover = HoverTool(    tooltips="""
    <div>
        <div>
        
        </div>
            <span style="font-size: 17px; font-weight: bold; color: #c51b8a; ">@desc</span>
        </div>   
        
            <table>
            <tr>
            <td><img src="@imgs" width="200" /></td>
            <td><img src="@spectra" width="200" /></td>
            </tr>
            </table>
            
            
        <div>
            <span style="font-size: 15px;">Location</span>
            <span style="font-size: 10px; font-weight: bold; color: #8856a7;">($x, $y )</span>
        </div>
        
     
        
        </div>
            <span style="font-size: 12px; font-weight: bold;">Confused source (if=1) = @confused</span>
        </div>  
        
    </div>
    """
)
        #<div>
        #<span style='font-size: 12px;'>Distance: @x</span>
        #</div>

        
# Define figure size, assign tools and give name
p = figure(plot_width=700, plot_height=700, tools=[hover, "pan,wheel_zoom,box_zoom,reset"], 
           title="The HI Parkes All Sky Survey", toolbar_location="above")

p.xaxis.axis_label = 'Distance [Mpc]'
p.yaxis.axis_label = 'log HI Mass'
p.xaxis.axis_label_text_font_size = "15pt"
p.yaxis.axis_label_text_font_size = "15pt"
p.title.text_font_size = '18pt'
p.xaxis.major_label_text_font_size = "15pt"
p.yaxis.major_label_text_font_size = "15pt"
#colors = ['black', 'red']


#Use the field name of the column source
mapper = linear_cmap(field_name='z', palette=viridis(8) ,low=min(df['W20max']) ,high=max(df['W20max']))

# Plot x and y data
p.scatter('x', 'y', size=14,  line_color=mapper,color=mapper,  source=source, fill_alpha=0.7)

color_bar = ColorBar(color_mapper=mapper['transform'], width=18,  location=(-2,-1), title='W20max')


p.add_layout(color_bar, 'right')

#p.line([8.5,9,10,11], [8.5,9,10,11], line_width=1, line_color='blue')

#p.yaxis.axis_label = 'HI mass observed [Mo]'
#p.xaxis.axis_label = 'HI mass expected [Mo]'

p.axis.major_tick_out = 0
p.axis.major_tick_in = 12
p.axis.minor_tick_in = 6
p.axis.minor_tick_out = 0

# Show in notebook
show(p)

# Save as html file and then open in browser to visualize
output_file('HIPASS_interactive_search.html', mode='inline')
save(p)

