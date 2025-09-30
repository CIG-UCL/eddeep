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
from external import voxelmorph

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))       
        
parser = argparse.ArgumentParser(description="Training script for the image translation part of Eddeep.")

# training and validation data, pre-trained translator
parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input 4D DW data to be corrected or to a folder containing them.')
parser.add_argument('-b', '--bvals', type=str, required=True, help="Path to the b-value file (in FSL style). There must be b-values strictly equal to 0!")
parser.add_argument('-tr', '--trans', type=str, required=True, help="Path to the pre-trained image translation model.")
parser.add_argument('-reg', '--reg', type=str, required=True, help="Path to the pre-trained image registration model.")
parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output corrected 4D DW data or to a folder containing them.')
parser.add_argument('-ot', '--output_trans', type=str, required=False, default=None, help='Path to the output corrected translated 4D DW data or to a folder containing them.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

out_trans = args.output_trans is not None

if os.path.isdir(args.input):
    os.makedirs(args.output, exist_ok=True)
    inputs = glob.glob(os.path.join(args.input, '*'))
else:
    inputs = [args.input]
      
#%% 

def preproc_img(dw, input_shape, int_norm=True):
    
    dw = sitk.Cast(dw, sitk.sitkFloat32)
    dw = sitk.Clamp(dw, lowerBound=0.0)
    dw = eddeep.utils.pad_image(dw, out_size=np.flip(input_shape))
    dw = sitk.GetArrayFromImage(dw)[np.newaxis,..., np.newaxis]
    if int_norm:
        dw = eddeep.utils.normalize_intensities_q(dw, 0.999)

    return tf.constant(dw)
    
@tf.function
def infer_translator(tensor):
    return translator(tensor, training=False)

@tf.function
def infer_registrator(b0_trans, dw_trans):
    return registrator([b0_trans, dw_trans], training=False)

warp_layer = voxelmorph.layers.SpatialTransformer(interp_method="linear", indexing="ij")
jac_layer = eddeep.layers.JacobianMultiplyIntensities(indexing='ij', is_shift=True)
@tf.function
def apply_corr(b0_trans, dw_trans, dw):

    dw = tf.cast(dw, tf.float32)

    transfo_estimator = tf.keras.Model(inputs=registrator.inputs,
                                       outputs=registrator.get_layer("compose_transfos").output)   
    full_transfo = transfo_estimator([b0_trans, dw_trans], training=False)
    
    dw_corr = warp_layer([dw, full_transfo])
    dw_corr = jac_layer([dw_corr, full_transfo])

    return [dw_corr, full_transfo]

    
translator = tf.keras.models.load_model(args.trans)
translator.trainable = False 

registrator = tf.keras.models.load_model(args.reg)
registrator.trainable = False 

input_shape = translator.input_shape[1:-1]

bvals = np.loadtxt(args.bvals)
ind_first_b0 = int(np.where(bvals == 0)[0][0])

for i in range(len(inputs)):
    
    dws_img = sitk.ReadImage(inputs[i])
    img_shape = dws_img.GetSize()[:-1]
    
    b0_img = dws_img[..., ind_first_b0]
    b0 = preproc_img(b0_img, input_shape)
    b0_trans = infer_translator(b0)
    
    dws_corr = []
    dws_corr_trans = []
    for j in trange(dws_img.GetSize()[-1], desc='img ' + str(i+1) + '/' + str(len(inputs))):
        
        dw = preproc_img(dws_img[..., j], input_shape)
        dw_trans = infer_translator(dw)  
        before = np.mean((b0_trans-dw_trans)**2)
        dw = preproc_img(dws_img[..., j], input_shape, int_norm=False)
        
        dw_corr_trans = infer_registrator(b0_trans, dw_trans)
        after = np.mean((b0_trans-dw_corr_trans)**2)
        
        if j == ind_first_b0:
            dw_corr = dws_img[..., j]    
        else:
            dw_corr, transfo = apply_corr(b0_trans, dw_trans, dw)           
            dw_corr = sitk.GetImageFromArray(dw_corr[0,...,0])
            dw_corr = eddeep.utils.unpad_image(dw_corr, img_shape)
            dw_corr = sitk.Cast(dw_corr, b0_img.GetPixelID())
            dw_corr.CopyInformation(b0_img)

        if out_trans:
            dw_corr_trans = sitk.GetImageFromArray(dw_corr_trans[0,...,0])
            dw_corr_trans = eddeep.utils.unpad_image(dw_corr_trans, img_shape)
            dw_corr_trans = sitk.Cast(dw_corr_trans, sitk.sitkFloat32)
            dw_corr_trans.CopyInformation(b0_img)

            dws_corr_trans.append(dw_corr_trans)
        dws_corr.append(dw_corr)
        
        print('vol: ',j,', bval: ',bvals[j],', before: ',before,', after',after)
        
    dws_corr = sitk.JoinSeries(dws_corr)
    dws_corr.CopyInformation(dws_img)
    if out_trans:
        dws_corr_trans = sitk.JoinSeries(dws_corr_trans)
        dws_corr_trans.CopyInformation(dws_img)
    
    if os.path.isdir(args.input):
        _, file_name = os.path.split(inputs[i]) 
        out_file = os.path.join(args.output, file_name)
        sitk.WriteImage(dws_corr, out_file)
    else:
        sitk.WriteImage(dws_corr, args.output)

    if out_trans:    
        if os.path.isdir(args.input):
            _, file_name = os.path.split(inputs[i]) 
            out_file = os.path.join(args.output_trans, file_name)
            sitk.WriteImage(dws_corr_trans, out_file)
        else:
            sitk.WriteImage(dws_corr_trans, args.output_trans)
