#%%
import os
import sys
eddeep_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(eddeep_dir)
import glob
import numpy as np
import tensorflow as tf     
import argparse
import SimpleITK as sitk
from tqdm import trange

import eddeep

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))       
        
parser = argparse.ArgumentParser(description="Training script for the image translation part of Eddeep.")

# training and validation data, pre-trained translator
parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input 4D DW data to be translated or to a folder containing them.')
parser.add_argument('-tr', '--trans', type=str, required=True, help="Path to the pre-trained image translation model.")
parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output translated 4D DW data or to a folder containing them.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

if os.path.isdir(args.input):
    os.makedirs(args.output, exist_ok=True)
    inputs = glob.glob(os.path.join(args.input, '*'))
else:
    inputs = [args.input]
      
#%% 

def preproc_img(dw, input_shape, norm_int=True, pad=True):
    
    dw = sitk.Cast(dw, sitk.sitkFloat32)
    dw = sitk.Clamp(dw, lowerBound=0.0)
    dw = eddeep.utils.pad_image(dw, out_size=np.flip(input_shape))
    dw = sitk.GetArrayFromImage(dw)[np.newaxis,..., np.newaxis]
    dw = eddeep.utils.normalize_intensities_q(dw, 0.999)

    return tf.constant(dw)
    
@tf.function
def infer_translator(tensor):
    return translator(tensor)
    
translator = tf.keras.models.load_model(args.trans)
translator.trainable = False 

input_shape = translator.input_shape[1:-1]

for i in range(len(inputs)):
    
    dws_img = sitk.ReadImage(inputs[i])
    img_shape = dws_img.GetSize()[:-1]

    dws_trans = []
    for j in trange(dws_img.GetSize()[-1], desc='img ' + str(i+1) + '/' + str(len(inputs))):

        dw = preproc_img(dws_img[..., j], input_shape)
        dw_trans = infer_translator(dw)
        
        dw_trans = sitk.GetImageFromArray(dw_trans[0,...,0])
        dw_trans = eddeep.utils.unpad_image(dw_trans, img_shape)
        
        dws_trans.append(dw_trans)
        
    dws_trans = sitk.JoinSeries(dws_trans)
    dws_trans.CopyInformation(dws_img)
    
    if os.path.isdir(args.input):
        _, file_name = os.path.split(inputs[i]) 
        out_file = os.path.join(args.output, file_name)
        sitk.WriteImage(dws_trans, out_file)
    else:
        sitk.WriteImage(dws_trans, args.output)
        
    
