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
parser.add_argument('-v', '--verbose', action='store_true', help='Print the b-value and the mean squared error to the b0 before and after correction for each volume.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

out_trans = args.output_trans is not None

if os.path.isdir(args.input):
    os.makedirs(args.output, exist_ok=True)
    if out_trans:
        os.makedirs(args.output_trans, exist_ok=True)
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

@tf.function
def estimate_transfo(b0_trans, dw_trans):
    return transfo_estimator([b0_trans, dw_trans], training=False)

def get_corrector():
    # layers are built for a fixed image shape, which varies across inputs on the native grid
    warp_layer = voxelmorph.layers.SpatialTransformer(interp_method="linear", indexing="ij")
    jac_layer = eddeep.layers.JacobianMultiplyIntensities(indexing='ij', is_shift=True)
    @tf.function
    def apply_corr(dw, full_transfo):

        dw = tf.cast(dw, tf.float32)
        dw_corr = warp_layer([dw, full_transfo])
        dw_corr = jac_layer([dw_corr, full_transfo])

        return dw_corr

    return apply_corr

def to_net_grid(img):
    return img if vox_size is None else eddeep.utils.change_img_res(img, [vox_size]*3)

def to_native_grid(img, native_img):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(native_img)
    resampler.SetUseNearestNeighborExtrapolator(True)
    return resampler.Execute(img)

def field_to_native(full_transfo, net_img, native_img):
    field = sitk.GetImageFromArray(np.asarray(full_transfo[0], dtype=np.float64), isVector=True)
    field = eddeep.utils.unpad_image(field, net_img.GetSize())
    field.CopyInformation(net_img)
    field = sitk.GetArrayFromImage(to_native_grid(field, native_img))
    # shifts are in voxels of the network grid, (z,y,x) order
    field *= vox_size / np.array(native_img.GetSpacing())[::-1]
    return tf.constant(field[np.newaxis], dtype=tf.float32)


translator = tf.keras.models.load_model(args.trans)
translator.trainable = False

registrator = tf.keras.models.load_model(args.reg)
registrator.trainable = False
transfo_estimator = tf.keras.Model(inputs=registrator.inputs,
                                   outputs=registrator.get_layer("compose_transfos").output)

input_shape = translator.input_shape[1:-1]
vox_size = eddeep.utils.get_vox_size(registrator)
if vox_size != eddeep.utils.get_vox_size(translator):
    sys.exit('The translator and the registrator were not trained at the same voxel size.')

bvals = np.loadtxt(args.bvals)
ind_first_b0 = int(np.where(bvals == 0)[0][0])

for i in range(len(inputs)):

    dws_img = sitk.ReadImage(inputs[i])
    img_shape = dws_img.GetSize()[:-1]

    b0_img = dws_img[..., ind_first_b0]
    b0_net = to_net_grid(b0_img)
    apply_corr = get_corrector()
    b0 = preproc_img(b0_net, input_shape)
    b0_trans = infer_translator(b0)

    dws_corr = []
    dws_corr_trans = []
    for j in trange(dws_img.GetSize()[-1], desc='img ' + str(i+1) + '/' + str(len(inputs))):

        dw_img = dws_img[..., j]
        dw = preproc_img(to_net_grid(dw_img), input_shape)
        dw_trans = infer_translator(dw)

        if out_trans or args.verbose:
            dw_corr_trans = infer_registrator(b0_trans, dw_trans)
        if args.verbose:
            before = np.mean((b0_trans-dw_trans)**2)
            after = np.mean((b0_trans-dw_corr_trans)**2)

        if j == ind_first_b0:
            dw_corr = dws_img[..., j]
        else:
            full_transfo = estimate_transfo(b0_trans, dw_trans)
            if vox_size is None:
                dw_corr = apply_corr(preproc_img(dw_img, input_shape, int_norm=False), full_transfo)
                dw_corr = sitk.GetImageFromArray(dw_corr[0,...,0])
                dw_corr = eddeep.utils.unpad_image(dw_corr, img_shape)
            else:
                dw = sitk.Clamp(sitk.Cast(dw_img, sitk.sitkFloat32), lowerBound=0.0)
                dw = sitk.GetArrayFromImage(dw)[np.newaxis,..., np.newaxis]
                dw_corr = apply_corr(dw, field_to_native(full_transfo, b0_net, b0_img))
                dw_corr = sitk.GetImageFromArray(dw_corr[0,...,0])
            dw_corr = sitk.Cast(dw_corr, b0_img.GetPixelID())
            dw_corr.CopyInformation(b0_img)

        if out_trans:
            dw_corr_trans = sitk.GetImageFromArray(dw_corr_trans[0,...,0])
            dw_corr_trans = eddeep.utils.unpad_image(dw_corr_trans, b0_net.GetSize())
            dw_corr_trans = sitk.Cast(dw_corr_trans, sitk.sitkFloat32)
            dw_corr_trans.CopyInformation(b0_net)
            if vox_size is not None:
                dw_corr_trans = to_native_grid(dw_corr_trans, b0_img)

            dws_corr_trans.append(dw_corr_trans)
        dws_corr.append(dw_corr)

        if args.verbose:
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

