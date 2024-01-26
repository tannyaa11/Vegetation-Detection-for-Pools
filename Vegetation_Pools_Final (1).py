#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import imageio
import warnings
warnings.filterwarnings('ignore')
import rasterio
import numpy
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from pathlib import Path
from PIL import Image


# In[3]:


#The images were extracted by taking snippets of area, 6.2 * 10.5 units
# Path to the folder containing PNG images
png_folder = (r"C:\Users\Tanya\Desktop\PENTAIR\Vegetation_Sample")

# Path to the folder where TIFF images will be saved
tif_folder = (r"C:\Users\Tanya\Desktop\PENTAIR\Vegetation_TIFF_images")

# Create the TIFF folder if it doesn't exist
os.makedirs(tif_folder, exist_ok=True)

# Convert PNG images to TIFF
for file_name in os.listdir(png_folder):
    if file_name.endswith('.png'):
        # Load the PNG image
        image_path = os.path.join(png_folder, file_name)
        image = imageio.imread(image_path)

        # Convert to TIFF format
        tiff_path = os.path.join(tif_folder, os.path.splitext(file_name)[0] + '.tif')
        imageio.imwrite(tiff_path, image, format='TIFF')

        print(f'Converted {file_name} to {os.path.basename(tiff_path)}')


# In[4]:


#eliminate warnings, if any
warnings.filterwarnings('ignore')

# Path to the folder containing TIFF images
folder_path = (r"C:\Users\Tanya\Desktop\PENTAIR\Vegetation_TIFF_images")

tif_files = list(Path(folder_path).glob('*.tif'))

# Iterate over each .tif file
for tif_file in tif_files:
    try:
        with rasterio.open(tif_file) as src:
            band_red = src.read(1)
            band_nir = src.read(2)
            
    # Further processing or analysis of the bands
    except rasterio.RasterioIOError as e:
        print(f"Error: {e}")


# In[5]:


#ignore divide by 0 errors
numpy.seterr(divide='ignore', invalid='ignore')

# NDVI
ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir + band_red)


# In[6]:


#print min and max NDVI  values
print(numpy.nanmin(ndvi))
print(numpy.nanmax(ndvi))


# In[7]:


# get the metadata of original GeoTIFF:
meta = src.meta
print(meta)

# get the dtype of our NDVI array:
ndvi_dtype = ndvi.dtype
print(ndvi_dtype)

# set the source metadata as kwargs we'll use to write the new data:
kwargs = meta

# update the 'dtype' value to match our NDVI array's dtype:
kwargs.update(dtype=ndvi_dtype)

# update the 'count' value since our output will no longer be a 4-band image:
kwargs.update(count=1)

# Finally, use rasterio to write new raster file 'data/ndvi.tif':
with rasterio.open('ndvi.tif', 'w', **kwargs) as dst:
        dst.write(ndvi, 1)
        


# In[8]:


from matplotlib import colors

#define range from -1 to 1
class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):

        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y), numpy.isnan(value))


# In[9]:


# Function to calculate NDVI
def calculate_ndvi(red_band, nir_band):
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return ndvi

# Function to convert NDVI to vegetation index
def convert_to_vegetation_index(ndvi):
    veg_index = np.where(ndvi > 0.5, 1, np.where(ndvi < -0.1, -1, 0))

    return veg_index

# Folder paths
input_folder = (r"C:\Users\Tanya\Desktop\PENTAIR\Vegetation_TIFF_images")
output_folder = (r"C:\Users\Tanya\Desktop\PENTAIR\NDVI_vegetation graph")

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all .tif files in the input folder
tif_files = [file for file in os.listdir(input_folder) if file.endswith(".tif")]

# Process each .tif file
for tif_file in tif_files:
    # Open the .tif file
    tif_path = os.path.join(input_folder, tif_file)
    with rasterio.open(tif_path) as src:
        red_band = src.read(3)  # Assuming red band is at index 3
        nir_band = src.read(4)  # Assuming NIR band is at index 4

        # Calculate NDVI
        ndvi = calculate_ndvi(red_band, nir_band)

        # Convert NDVI to vegetation index
        veg_index = convert_to_vegetation_index(ndvi)

        # Plot vegetation index on a graph
        plt.imshow(veg_index, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='Vegetation Index')
        plt.title('Vegetation Index Image')
        
        # Save the graph in the output folder
        output_file = os.path.join(output_folder, f"{tif_file[:-4]}.png")
        plt.savefig(output_file)
        
        # Show the graph (optional)
        plt.show()
    


# In[5]:


#green percentage calculation
#defining range of green pixels
def calculate_green_percentage(image_path):
    image = Image.open(image_path)
    width, height = image.size
    green_pixels = 0

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            
            # Check if the pixel is green
            if pixel[1] > pixel[0] and pixel[1] > pixel[2]:
                green_pixels += 1
    total_pixels = width * height
    green_percentage = (green_pixels / total_pixels) * 100
    return green_percentage

# Specify the path to the folder containing the .png images
folder_path = (r"C:\Users\Tanya\Desktop\PENTAIR\NDVI_vegetation graph")

# Get the list of all files in the folder
files = os.listdir(folder_path)

# Initialize lists to store the filenames and green percentages
filenames = []
green_percentages = []

# Iterate over each file in the folder
for file in files:
    # Check if the file is a .png image
    if file.endswith(".png"):
        image_path = os.path.join(folder_path, file)
        green_percentage = calculate_green_percentage(image_path)
        filenames.append(file)
        green_percentages.append(green_percentage)
        print(f"Green percentage of {file}: {green_percentage}%")
        
# Plot the bar graph
plt.bar(filenames, green_percentages)
plt.xlabel("Image")
plt.ylabel("Green Percentage")
plt.title("Green Percentage of Images")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()





# In[ ]:





# In[ ]:





# In[ ]:




