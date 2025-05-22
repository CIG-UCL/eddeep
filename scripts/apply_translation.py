import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import glob
import numpy as np
import tensorflow as tf     
import argparse
import eddeep
import SimpleITK as sitk

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))       
        
parser = argparse.ArgumentParser(description="Training script for the image translation part of Eddeep.")

# training and validation data, pre-trained translator
parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input 4D DW data to be translated or to a folder containing them.')
parser.add_argument('-tr', '--trans', type=str, required=True, help="Path to the pre-trained image translation model.")
parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output translated 4D DW data or to a folder containing them.')
parser.add_argument('-k', '--kpad', type=int, required=False, default=5, help='k to pad the input so that its shape is of form 2**k. Has to be >= number encoding steps.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

if os.path.isdir(args.input):
    os.makedirs(args.output, exist_ok=True)
    inputs = glob.glob(os.path.join(args.input, '*'))
else:
    inputs = [args.input]
      
#%% 

def preproc_img(img, kpad=5, norm_int=True, pad=True):
    
    if norm_int: 
        img = eddeep.utils.normalize_intensities(img, dtype=sitk.sitkFloat32)
    if pad: 
        img = eddeep.utils.pad_image(img, k=kpad)

    return tf.constant(sitk.GetArrayFromImage(img)[np.newaxis,..., np.newaxis])
    
@tf.function
def infer_translator(tensor):
    return translator(tensor)
    
translator = eddeep.models.pix2pix_gen.load(args.trans)
translator.trainable = False 

for i in range(len(inputs)):
    
    dws_img = sitk.ReadImage(inputs[i])
    dw0_img = dws_img[..., 0]
    vol_size = dws_img.GetSize()[:-1]
    
    dws_trans = []
    for j in range(dws_img.GetSize()[-1]):

        dw = preproc_img(dws_img[..., j], args.kpad)
        dw_trans = infer_translator(dw)
        
        dw_trans = sitk.GetImageFromArray(dw_trans[0,...,0])
        dw_trans = eddeep.utils.unpad_image(dw_trans, vol_size, k=args.kpad)
        dw_trans = sitk.Cast(dw_trans, sitk.sitkFloat32)
        dw_trans.CopyInformation(dw0_img)
        
        dws_trans.append(dw_trans)
        
        print('vol: ',j)
        
    dws_trans = sitk.JoinSeries(dws_trans)
    dws_trans.CopyInformation(dws_img)
    
    if os.path.isdir(args.input):
        _, file_name = os.path.split(inputs[i]) 
        out_file = os.path.join(args.output, file_name)
        sitk.WriteImage(dws_trans, out_file)
    else:
        sitk.WriteImage(dws_trans, args.output)
        
    
