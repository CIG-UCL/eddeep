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
parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input 4D DW data to be corrected or to a folder containing them.')
parser.add_argument('-b', '--bvals', type=str, required=True, help="Path to the b-value file (in FSL style). There must be b-values strictly equal to 0!")
parser.add_argument('-tr', '--trans', type=str, required=True, help="Path to the pre-trained image translation model.")
parser.add_argument('-reg', '--reg', type=str, required=True, help="Path to the pre-trained image registration model.")
parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output corrected 4D DW data or to a folder containing them.')
parser.add_argument('-ot', '--output_trans', type=str, required=False, default=None, help='Path to the output corrected translated 4D DW data or to a folder containing them.')
parser.add_argument('-g', '--bvecs', type=str, required=False, default=None, help="Path to the b-vector file (in FSL style). Required if -og is provided.")
parser.add_argument('-og', '--output_bvecs', type=str, required=False, default=None, help='Path to the output rotated b-vector file or to a folder containing them.')
parser.add_argument('-in', '--interp', type=str, required=False, default='linear', choices=['linear', 'spline'], help="Interpolation for the final resampling: 'linear' (trilinear) or 'spline' (cubic B-spline). Default: linear.")
parser.add_argument('-v', '--verbose', action='store_true', help='Print the b-value and the mean squared error to the b0 before and after correction for each volume.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
if args.output_bvecs is not None and args.bvecs is None:
    parser.error('-g/--bvecs is required when -og/--output_bvecs is provided.')

out_trans = args.output_trans is not None
out_bvecs = args.output_bvecs is not None
interp = {'linear': sitk.sitkLinear, 'spline': sitk.sitkBSpline}[args.interp]

if os.path.isdir(args.input):
    os.makedirs(args.output, exist_ok=True)
    if out_trans:
        os.makedirs(args.output_trans, exist_ok=True)
    if out_bvecs:
        os.makedirs(args.output_bvecs, exist_ok=True)
    inputs = glob.glob(os.path.join(args.input, '*'))
else:
    inputs = [args.input]

#%%

def preproc_img(dw, input_shape, vox_size, int_norm=True):

    dw = eddeep.utils.to_vox_size(dw, vox_size)
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


def to_native_grid(img, native_img):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(native_img)
    resampler.SetUseNearestNeighborExtrapolator(True)
    return resampler.Execute(img)

def field_net2phys(disp_field, net_img, native_img):

    disp_field = np.asarray(disp_field[0], dtype=np.float64)
    # shifts in voxels of the network grid, (z,y,x) order -> mm along the image axes, (x,y,z) order
    disp_field = sitk.GetImageFromArray(np.ascontiguousarray(disp_field[..., ::-1] * net_img.GetSpacing()), isVector=True)
    disp_field = eddeep.utils.unpad_image(disp_field, net_img.GetSize())
    disp_field.CopyInformation(net_img)
    if net_img is not native_img:
        disp_field = to_native_grid(disp_field, native_img)
    jac = sitk.Abs(sitk.DisplacementFieldJacobianDeterminant(disp_field))
    # mm along the image axes -> world displacements
    disp_world = sitk.GetImageFromArray(sitk.GetArrayFromImage(disp_field) @ np.reshape(native_img.GetDirection(), (3, 3)).T, isVector=True)
    disp_world.CopyInformation(disp_field)

    return disp_world, jac

def correct(dw_img, disp_field, jac, interp):

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(dw_img)
    resampler.SetTransform(sitk.DisplacementFieldTransform(disp_field))
    resampler.SetInterpolator(interp)
    resampler.SetUseNearestNeighborExtrapolator(True)
    dw = sitk.Clamp(sitk.Cast(dw_img, sitk.sitkFloat32), lowerBound=0.0)
    dw = sitk.Clamp(resampler.Execute(dw), lowerBound=0.0)

    return dw * sitk.Cast(jac, sitk.sitkFloat32)

def rotate_bvec(rigid, bvec, spacing, flip_x):
    # rigid is in array (z,y,x) voxel coordinates; bring the rotation to (x,y,z) physical axes
    rot = np.asarray(rigid)[0, :3, :3][::-1, ::-1]
    rot = np.diag(spacing) @ rot @ np.diag(1 / np.asarray(spacing))
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    # FSL bvecs have x flipped when the voxel-to-world matrix has a positive determinant
    flip = np.diag([-1, 1, 1]) if flip_x else np.eye(3)
    return flip @ rot.T @ flip @ bvec


translator = tf.keras.models.load_model(args.trans)
translator.trainable = False

registrator = tf.keras.models.load_model(args.reg)
registrator.trainable = False
transfo_estimator = tf.keras.Model(inputs=registrator.inputs,
                                   outputs=[registrator.get_layer("compose_transfos").output,
                                            registrator.get_layer("build_rigid_transfo").output])

input_shape = translator.input_shape[1:-1]
vox_size = eddeep.utils.get_vox_size(registrator)
if vox_size != eddeep.utils.get_vox_size(translator):
    sys.exit('The translator and the registrator were not trained at the same voxel size.')

bvals = np.loadtxt(args.bvals)
ind_first_b0 = int(np.where(bvals == 0)[0][0])
if out_bvecs:
    bvecs = np.loadtxt(args.bvecs)

for i in range(len(inputs)):

    dws_img = sitk.ReadImage(inputs[i])
    b0_img = dws_img[..., ind_first_b0]
    b0_net = eddeep.utils.to_vox_size(b0_img, vox_size)
    resampled = b0_net is not b0_img
    b0 = preproc_img(b0_net, input_shape, vox_size)
    b0_trans = infer_translator(b0)
    if out_bvecs:
        bvecs_rot = bvecs.copy()
        flip_x = np.linalg.det(np.reshape(b0_img.GetDirection(), (3, 3))) > 0

    dws_corr = []
    dws_corr_trans = []
    for j in trange(dws_img.GetSize()[-1], desc='img ' + str(i+1) + '/' + str(len(inputs))):

        dw_img = dws_img[..., j]
        dw = preproc_img(dw_img, input_shape, vox_size)
        dw_trans = infer_translator(dw)

        if out_trans or args.verbose:
            dw_corr_trans = infer_registrator(b0_trans, dw_trans)
        if args.verbose:
            before = np.mean((b0_trans-dw_trans)**2)
            after = np.mean((b0_trans-dw_corr_trans)**2)

        if j == ind_first_b0:
            dw_corr = dws_img[..., j]
        else:
            disp_field, rigid = estimate_transfo(b0_trans, dw_trans)
            if out_bvecs:
                bvecs_rot[:, j] = rotate_bvec(rigid, bvecs[:, j], b0_net.GetSpacing(), flip_x)
            disp_field, jac = field_net2phys(disp_field, b0_net, b0_img)
            dw_corr = correct(dw_img, disp_field, jac, interp)
            dw_corr = sitk.Cast(dw_corr, b0_img.GetPixelID())
            dw_corr.CopyInformation(b0_img)

        if out_trans:
            dw_corr_trans = sitk.GetImageFromArray(dw_corr_trans[0,...,0])
            dw_corr_trans = eddeep.utils.unpad_image(dw_corr_trans, b0_net.GetSize())
            dw_corr_trans = sitk.Cast(dw_corr_trans, sitk.sitkFloat32)
            dw_corr_trans.CopyInformation(b0_net)
            if resampled:
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

    if out_bvecs:
        if os.path.isdir(args.input):
            _, file_name = os.path.split(inputs[i])
            file_name = file_name.split('.nii')[0] + '.bvec'
            np.savetxt(os.path.join(args.output_bvecs, file_name), bvecs_rot, fmt='%.6f')
        else:
            np.savetxt(args.output_bvecs, bvecs_rot, fmt='%.6f')
