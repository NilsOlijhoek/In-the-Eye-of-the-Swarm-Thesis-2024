import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from glob import glob

use_GPU = core.use_gpu()
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[use_GPU]}')

# model name and path

#Custom model path, needs to include model file:

model_path = "C:\\Users\\Admin\\.cellpose\\models\\Model" #@param {type:"string"}

#Path to image folder:

dir = "C:\\Users\\Admin\\Fluorscent_Images" #@param {type:"string"}

#Channel Parameters:

Channel_to_use_for_segmentation = "Grayscale" #@param ["Grayscale", "Blue", "Green", "Red"]

#  If you have a secondary channel that can be used, choose it here:

Second_segmentation_channel= "None" #@param ["None", "Blue", "Green", "Red"]


# Match the channel to number
if Channel_to_use_for_segmentation == "Grayscale":
  chan = 0
elif Channel_to_use_for_segmentation == "Blue":
  chan = 3
elif Channel_to_use_for_segmentation == "Green":
  chan = 2
elif Channel_to_use_for_segmentation == "Red":
  chan = 1


if Second_segmentation_channel == "Blue":
  chan2 = 3
elif Second_segmentation_channel == "Green":
  chan2 = 2
elif Second_segmentation_channel == "Red":
  chan2 = 1
elif Second_segmentation_channel == "None":
  chan2 = 0

#Segmentation parameters:

# diameter of cells (set to zero to use diameter from training set):
diameter =  0 #@param {type:"number"}
# threshold on flow error to accept a mask (set higher to get more cells, e.g. in range from (0.1, 3.0), OR set to 0.0 to turn off so no cells discarded):
flow_threshold = 1 #@param {type:"slider", min:0.0, max:3.0, step:0.1}
# threshold on cellprob output to seed cell masks (set lower to include more pixels or higher to include fewer, e.g. in range from (-6, 6)):
cellprob_threshold=0 #@param {type:"slider", min:-6, max:6, step:1}

# gets image files in dir (ignoring image files ending in _masks)
files = io.get_image_files(dir, '_masks')
print(files)
images = [io.imread(f) for f in files]

# declare model
model = models.CellposeModel(gpu=True, 
                             pretrained_model=model_path)

# use model diameter if user diameter is 0
diameter = model.diam_labels if diameter==0 else diameter

# run model on test images
masks, flows, styles = model.eval(images, 
                                   channels=[chan, chan2],
                                   diameter=diameter,
                                   flow_threshold=flow_threshold,
                                   cellprob_threshold=cellprob_threshold
                                   )
from cellpose import io

io.masks_flows_to_seg(images, 
                      masks, 
                      flows, 
                      diameter*np.ones(len(masks)), 
                      files, 
                      channels=[chan, chan2])

io.save_masks(images, 
              masks, 
              flows, 
              files, 
              channels=[chan, chan2],
              png=True, # save masks as PNGs and save example image
              tif=True, # save masks as TIFFs
              save_txt=True, # save txt outlines for ImageJ
              save_flows=True, # save flows as TIFFs
              save_outlines=False, # save outlines as TIFFs 
              )
    
