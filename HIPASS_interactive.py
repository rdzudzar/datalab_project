
# coding: utf-8

# In[62]:


__author__ = 'Robert Dzudzar <robertdzudzar@gmail.com>, <rdzudzar@swin.edu.au>'
__version__ = '20190511' # yyyymmdd; version datestamp of this notebook
#__datasets__ = ['des_dr1']
__keywords__ = ['Interactive', 'Neutral Hydrogen', 'Spectra', 'Galaxies','Bokeh']


# # Interactively examining the HI Parkes All Sky Survey (HIPASS)
# *Robert Dzudzar*

# ### Table of contents
# * [Goals & notebook summary](#goals)
# * [Disclaimer & Attribution](#attribution)
# * [Imports & setup](#import)
# * [Authentication](#auth)  
# 
# * #### [Import HIPASS data](#chapter1)
#     * [Plot the Sky coverage of the HIPASS survey](#chapter1.1)
#     * [Choose dataset to visuelise](#chapter1.2)
#     * [Scraping url-s where the data of the HIPASS spectra is storred](#chapter1.3)
#     * [Creating list of HIPASS sources](#chapter1.4)
#     * [Extracting spectral information from HIPASS database](#chapter1.5)
#     * [Plotting the HI spectra for each source](#chapter1.6)  
# 
# * #### [Query optical counterparts of the HIPASS sources](#chapter2)
#     * [Create a list of coordinates](#chapter2.1)
#     * [Download and save images from SkyView](#chapter2.2)  
# 
# * #### [Setting up Bokeh for interactive examination](#chapter3)
#     * [Extracting/Sorting images and spectra which we have](#chapter3.1)
#     * [Approximate Distance and HI mass](#chapter3.2)  
# 
# * #### [Interactive visualization with Bokeh](#chapter4)  
# 
# 
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

# In[119]:


# std lib
from getpass import getpass
import os
from os import listdir
import pathlib
from urllib.error import HTTPError

# 3rd party
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import binned_statistic_2d
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import html5lib
import requests
from bs4 import BeautifulSoup

# astropy
from astropy.table import Table
from astropy import utils, io, convolution, stats
from astropy.visualization import make_lupton_rgb
from astropy import coordinates, units as u, wcs
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView

 # bokeh
from bokeh.io import output_notebook
from bokeh.palettes import BuGn8, viridis
from bokeh.transform import linear_cmap
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.plotting import figure, output_file, show, ColumnDataSource, gridplot, save
from bokeh.models import HoverTool, BoxSelectTool
output_notebook()


# Data Lab
from dl import queryClient as qc
# Data Lab
from dl import authClient as ac, queryClient as qc, storeClient as sc, helpers


# In[105]:


# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

# Either get token for anonymous user
token = ac.login('anonymous')

# ... or for authenticated user
#token = ac.login(input("Enter user name: "),getpass("Enter password: "))


# <a class="anchor" id="chapter1"></a>
# # Import HIPASS data

# In[248]:


# Load galaxy properties from HIPASS data (https://ui.adsabs.harvard.edu/abs/2004MNRAS.350.1195M/abstract)
HIPASS_data = Table.read('HIPASS_catalog.fit')
df_hipass = HIPASS_data.to_pandas()

# Display the dataframe
df_hipass
# You can see all the columns:
#df_hipass.columns


# <a class="anchor" id="chapter1.1"></a>
# ## Plot the Sky coverage of the HIPASS survey

# In[108]:


# Plot HIPASS survey

fig = plt.figure(figsize=(14,14)) 
ax = fig.add_subplot(111, projection="mollweide") # Using mollweide projection
# Converting RA and DEC from deg to radians
im = ax.scatter(np.radians(df_hipass['_RAJ2000']-180), np.radians(df_hipass['_DEJ2000']), c=df_hipass['RVmom'], cmap='viridis', s=20)
# Adding colorbar for the sources, based on their Velocity
cb = plt.colorbar(im, orientation = 'horizontal', shrink = 0.8)
# RVmom ==> Flux-weighted mean velocity of profile clipped at RVlo and RVhi (explained in the online HIPASS table)
cb.set_label(r'RVmom [km s$^{-1}$]', size=18) 


ax.set_xlabel(r'$\mathrm{RA[\degree]}$')
ax.xaxis.label.set_fontsize(20)
ax.set_ylabel(r'$\mathrm{Dec[\degree]}$')
ax.yaxis.label.set_fontsize(20)

# Invalid value encountered probably because of x limits are +/- Pi which are both singularities on the Mollweide projection.


# <a class="anchor" id="chapter1.2"></a>
# # Choose dataset to visualise
# ### Type 'True' for the selected dataset

# In[226]:


# The 100 most HI massive galaxies from HIPASS  
the_100_most_massive = 'True'

# The 100 least HI massive galaxies from HIPASS  
the_100_least_massive = 'False'

# The 100 confused sources from HIPASS  
the_100_confused = 'False'


# ### Set-up saving folder

# In[227]:


# Check needed directories and create them if they dont exist. In these directories HI spectra and optical images will be saved.
# Check condition of which dataset was chosen and create respective path if they don't exist.

if the_100_most_massive == 'True':
    pathlib.Path('./HIPASS_spectra_most_massive').mkdir(parents=True, exist_ok=True) 
    pathlib.Path('./HIPASS_images_most_massive').mkdir(parents=True, exist_ok=True) 
    spectra_path = './HIPASS_spectra_most_massive/' # Will be used for folder to save spectra
    images_path = './HIPASS_images_most_massive/' # Will be used for folder to save images
    interactive = 'most_massive' # Will be used to save hmtl file.
    
elif the_100_least_massive == 'True':
    spectra_path = pathlib.Path('./HIPASS_spectra_least_massive').mkdir(parents=True, exist_ok=True) 
    images_path = pathlib.Path('./HIPASS_images_least_massive').mkdir(parents=True, exist_ok=True) 
    spectra_path = './HIPASS_spectra_least_massive/'
    images_path = './HIPASS_images_least_massive/'
    interactive = 'least_massive'
    
elif the_100_confused == 'True':
    spectra_path = pathlib.Path('./HIPASS_spectra_confused').mkdir(parents=True, exist_ok=True) 
    images_path = pathlib.Path('./HIPASS_images_confused').mkdir(parents=True, exist_ok=True) 
    spectra_path = './HIPASS_spectra_confused/'
    images_path = './HIPASS_images_confused/'
    interactive = 'confused'
    
else:
    print("There is an error in selection of the dataset")


# ### Approximate HI masses and Distances

# In[228]:


H0 = 70 # Hubble constant
# Add distance HI mass to the table
# These are created by addopting RVmom as the recessional velocity for distance (RVmom * H0) and mass estimation! 

df_hipass['logHI_mass_approx'] = pd.Series(np.log10(2.365*10e5*((df_hipass['RVmom']/H0)**2)*df_hipass['Sint']), index=df_hipass.index)
df_hipass['Distance_approx'] = pd.Series( (df_hipass['RVmom']/H0), index=df_hipass.index)

if the_100_most_massive == 'True':
        
    # Sort original HIPASS dataframe by the HI mass column - from highest to lowest and reset indexing so that we select first 100
    df_by_mass_dsc = df_hipass.sort_index(by='logHI_mass_approx', ascending=False).reset_index()
    df_most_HI_massive = df_by_mass_dsc[0:100]
    df = df_most_HI_massive
    
elif the_100_least_massive=='True':
    # Sort original HIPASS dataframe by the HI mass column - from highest to lowest and reset indexing so that we select first 100
    df_by_mass_asc = df_hipass.sort_index(by='logHI_mass_approx', ascending=True).reset_index()
    df_least_HI_massive = df_by_mass_asc[0:100]
    df = df_least_HI_massive
    
elif the_100_confused == 'True':
    # Sort original HIPASS dataframe by the HI mass column - from highest to lowest and reset indexing so that we select first 100
    df_conf = df_hipass.sort_index(by='cf', ascending=False).reset_index()
    df_confused = df_conf[0:100]
    df = df_confused
    
else:
    print('Error: You either did not selected the dataset or you selected multiple datasets')
    df = 'None'


# In[229]:


df


# <a class="anchor" id="chapter1.3"></a>
# ## Scraping url-s where the data of the HIPASS spectra is storred

# In[232]:


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


# <a class="anchor" id="chapter1.4"></a>
# ##  Creating list of HIPASS sources

# In[234]:


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

# In[235]:


# We want to go to each url and extract only the spectra data
# From each url we need Intensity, Velocity and Channel information

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

# In[236]:


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
    
    fig.savefig(spectra_path+'{0}'.format(idx), overwrite=True)
    plt.close(fig)
    
    #plt.show()       


# <a class="anchor" id="chapter2"></a>
# # Query optical counterparts of the HIPASS sources

# In[246]:


# Using SkyView to get the DSS images of the sources
# list all available image data which can be obtained from SkyView:
#SkyView.list_surveys() 

# For DSS: 
#'Optical:DSS': ['DSS',
#                  'DSS1 Blue',
#                  'DSS1 Red',
#                  'DSS2 Red',
#                  'DSS2 Blue',
#                  'DSS2 IR'],


# <a class="anchor" id="chapter2.1"></a>
# ## Create a list of coordinates

# In[241]:


# To query Sky position (images) of sources, we need central position of each detection in HIPASS so we extract them using SkyCoord
 
c = []
for each_galaxy in df.index:
    center = SkyCoord(df['_RAJ2000'][each_galaxy], df['_DEJ2000'][each_galaxy], frame='icrs', unit="deg")
    c.append(center)
    #print(center)


# <a class="anchor" id="chapter2.2"></a>
# ## Download and save images from SkyView
# ### (This cell takes long to run for large number of sources (~1h for 500))

# In[242]:


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
        images = SkyView.get_images(position=center, pixels=[1000,1000], survey=Survey, radius=7.5*u.arcmin)
        
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
    
        #fig.savefig('./HIPASS_images/{0}.png'.format(idx), overwrite=True)
        
        fig.savefig(images_path+'{0}'.format(idx), overwrite=True)
        #plt.show() 
        plt.close(fig)
        
    except HTTPError:
        print('Image not found in the {0} filter'.format(Survey))
        continue
    
    
    #ax.axis([100, 200, 100, 200])


# <a class="anchor" id="chapter3"></a>
# # Setting up Bokeh for interactive examination

# <a class="anchor" id="chapter3.1"></a>
# ### Extracting/Sorting images and spectra which we have

# In[244]:


# Check files in the downloaded folder and play around with the strings to sort them and extract

extension = '.png'
mypath_images = images_path # Either images_path or spectra_path since the numbering and length should be the same.

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
    New_list_s = spectra_path+str(i)+'.png'
    New_list_i = images_path+str(i)+'.png'
    new_list_sorted.append(New_list_s)
    new_list_images.append(New_list_i)
#print(new_list_sorted)


# <a class="anchor" id="chapter4"></a>
# # Interactive visualization with Bokeh

# In[245]:


# Add bokeh features
# We are plotting x and y data
# As desc - description - we will have name of the object
# as spectra and imgs -- we will have spectrum image and optical image for each source as we hover above plotted points
# Depending on the speed of your internet, when first time hovering on points - wait a couple of seconds for images to appear


source = ColumnDataSource(
        data=dict(
            x = df['Distance_approx'],
            y = df['logHI_mass_approx'], 
            z = df['W20max'],
            desc = HIPASS_sources ,
            confused = df['cf'],
            Int = df['Sint'],
            spectra = new_list_sorted,
            imgs = new_list_images,))

# Adding html code to say how the images, spectra and other information will be displayed when one hover on points
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
output_file('HIPASS_interactive_{0}_search.html'.format(interactive), mode='inline')
save(p)


# <a class="anchor" id="resources"></a>
# # Resources and references

# #### The HIPASS data 
# Barnes et al (http://adsabs.harvard.edu/abs/2001MNRAS.322..486B)  
# Data used in this notebook are published by Meyer et al. 2004 https://ui.adsabs.harvard.edu/?#abs/2004MNRAS.350.1195M  
# 
# ##### The HIPASS table 
# Table is obtained through VizieR services: This research has made use of the VizieR catalogue access tool, CDS,  Strasbourg, France (DOI : 10.26093/cds/vizier). The original description of the VizieR service was published in A&AS 143, 23
# 
# The Parkes telescope is part of the Australia Telescope which is funded by the Commonwealth of Australia for operation as a National Facility managed by CSIRO. The full HI database is located here http://www.atnf.csiro.au/research/multibeam/release/
# 
# #### SkyView 
# Sky View has been developed with generous support from the NASA AISR and ADP programs (P.I. Thomas A. McGlynn) under the auspices of the High Energy Astrophysics Science Archive Research Center (HEASARC) at the NASA/ GSFC Astrophysics Science Division. (https://skyview.gsfc.nasa.gov/current/cgi/survey.pl)
# 
# #### Optical images 
# Obtained from Original Digitised Data Service: STScI, ROE, AAO, UK-PPARC, CalTech, National Geographic Society. (http://archive.stsci.edu/dss/copyright.html)
# 
# #### Used Python3 and Python packages:
# 
# Astropy (Astropy Collaboration, doi: 10.1051/0004-6361/201322068) https://www.astropy.org/  
# Pandas The official documentation is hosted on PyData.org: https://pandas.pydata.org/pandas-docs/stable   
# Bokeh : Bokeh Development Team (2018). Bokeh: Python library for interactive visualization URL http://www.bokeh.pydata.org.  
# Matplotlib (Hunter el al. 2007, doi: 10.1109/MCSE.2007.55) http://matplotlib.org/  
# Numpy (van der Walt 2011, doi: 10.1109/MCSE.2011.37) http://www.numpy.org/  
# Requests (Copyright 2018 Kenneth Reitz), https://2.python-requests.org/en/master/  
# BeautifulSoup https://www.crummy.com/software/BeautifulSoup/bs4/doc/  
