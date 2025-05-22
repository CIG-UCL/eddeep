import os
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
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
parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input 4D DW data to be corrected or to a folder containing them.')
parser.add_argument('-b', '--bvals', type=str, required=True, help="Path to the b-value file (in FSL style). There must be b-values strictly equal to 0!")
parser.add_argument('-tr', '--trans', type=str, required=True, help="Path to the pre-trained image translation model.")
parser.add_argument('-reg', '--reg', type=str, required=True, help="Path to the pre-trained image registration model.")
parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output corrected 4D DW data or to a folder containing them.')
parser.add_argument('-ot', '--output_trans', type=str, required=False, default=None, help='Path to the output corrected translated 4D DW data or to a folder containing them.')
parser.add_argument('-k', '--kpad', type=int, required=False, default=5, help='k to pad the input so that its shape is of form 2**k. Has to be >= number encoding steps.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

out_trans = args.output_trans is not None

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
    return translator(tensor, training=False)

@tf.function
def infer_registrator(b0_trans, dw_trans, dw):
    return registrator.apply_corr(b0_trans, dw_trans, dw)

    
translator = eddeep.models.pix2pix_gen.load(args.trans)
translator.trainable = False 

registrator = eddeep.models.eddy_reg.load(args.reg)
registrator.trainable = False 

bvals = np.loadtxt(args.bvals)
ind_first_b0 = int(np.where(bvals == 0)[0][0])

for i in range(len(inputs)):
    
    dws_img = sitk.ReadImage(inputs[i])
    vol_size = dws_img.GetSize()[:-1]
    
    b0_img = dws_img[..., ind_first_b0]
    b0 = preproc_img(b0_img, args.kpad)
    b0_trans = infer_translator(b0)
    
    dws_corr = []
    dws_corr_trans = []
    for j in range(dws_img.GetSize()[-1]):
        

        dw = preproc_img(dws_img[..., j], args.kpad)
        dw_trans = infer_translator(dw)  
        before = np.mean((b0_trans-dw_trans)**2)
        dw = preproc_img(dws_img[..., j], args.kpad, norm_int=False)
        
        dw_corr_trans = registrator([b0_trans, dw_trans])
        after = np.mean((b0_trans-dw_corr_trans)**2)
        
        if j == ind_first_b0:
            dw_corr = dws_img[..., j]    
        else:
            dw_corr = infer_registrator(b0_trans, dw_trans, dw)           
            dw_corr = sitk.GetImageFromArray(dw_corr[0,...,0])
            dw_corr = eddeep.utils.unpad_image(dw_corr, vol_size, k=args.kpad)
            dw_corr = sitk.Cast(dw_corr, b0_img.GetPixelID())
            dw_corr.CopyInformation(b0_img)

        if out_trans:
            dw_corr_trans = sitk.GetImageFromArray(dw_corr_trans[0,...,0])
            dw_corr_trans = eddeep.utils.unpad_image(dw_corr_trans, vol_size, k=args.kpad)
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
