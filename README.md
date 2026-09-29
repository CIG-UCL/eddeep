# Eddeep

**Eddeep** is composed of 2 models in sequence:
  1) **Translator**: Restore correspondences between images.
  2) **Registrator**: Estimate the distortion and apply correction.

<p align="center">
<img src="imgs/diagram_eddeep.svg" width="85%">
</p>

## Installation

```bash
git clone git@github.com:CIG-UCL/eddeep.git
cd eddeep
pip install -r requirements.txt
```


## Training Eddeep

### Preprocessing

#### 1) Pre-correction with an external tool (for translator training only)
During training (but not at inference), the translator takes as input images that have been corrected for eddy distortions by an external tool. You can typically use [FSL Eddy](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/eddy(2f)UsersGuide.html) or [Tortoise](https://tortoise.nibib.nih.gov/tortoise) for that.

#### 2) Creation of the translation targets (for translator training only)
  - Choose a moderately high (700-3000) b-value among the acquired ones.
  - For each subject, average all the volumes for this b-value to obtain a direction average image\
    (assuming b-vectors are uniformly sampled on the sphere).

#### 3) Data organisation:
For the dataloader, the 4D DW data must be chopped into 3D volumes and organised according to the following nested structure: {subject_d} > {PED} > b{b-value} > {vol_gradDir}.nii.gz. The target image for translation following: {subject_d} > {PED} > {vol}_b{target b-value}_mean.nii.gz. For example:
```
├── sub_001
│   ├── AP
│   │   ├── b0
│   │   │   ├── vol_dir1.nii.gz
│   │   │   ├── vol_dir2.nii.gz
│   │   │   ├── ...
│   │   ├── b1000
│   │   │   ├── ...
│   │   ├── ...
│   │   ├── vol_b2000_mean.nii.gz (only for translation)
│   │   ├── ...
│   └── PA
│       ├── ...
├── sub_002
│   ├── ...
├── ...
```
  - For the translator, the input data is pre-corrected and there is a translation target.
  - For the registrator, the input data is the raw DW data.

There must be b=0!

### Training the translator
```bash
eddeep_dir=<path-to-eddeep>
model_dir=<path-to-models>
```

```bash
bvaltarget=<chose-target-bvalue>
data_precorr_train_dir=<path-to-precorrected-training-data-dir>
data_precorr_val_dir=<path-to-precorrected-validation-data-dir>

python ${eddeep_dir}/scripts/train_eddeep_trans.py -t ${data_precorr_train_dir}\
                                                   -v ${data_precorr_val_dir}\
                                                   -o ${model_dir}/trans\
                                                   -B ${bvaltarget} -e 400 -as 0.5 -ai 0.5\
                                                   -vs 2
```
Images are resampled to an isotropic voxel size (`-vs`, 2 mm by default, 0 to keep the native resolution), which is stored in the model. The registrator training and the inference scripts read it from the model, so it only needs to be set here.

### Training the registrator
```bash
data_train_dir=<path-to-training-data-dir>
data_val_dir=<path-to-validation-data-dir>

python ${eddeep_dir}/scripts/train_eddeep_corr.py -t ${data_train_dir}\
                                                  -v ${data_val_dir}\
                                                  -tr ${model_dir}/trans_gen_best.keras\
                                                  -o ${model_dir}/corr\
                                                  -p 1\
                                                  -e 200 -as 0.5
```

## Correct for eddy distortions with a pre-trained **Eddeep**
Given:
  - A pre-trained **Eddeep** translator (e.g. `trans_gen_best.keras`).
  - A pre-trained **Eddeep** registrator (e.g. `corr_best.keras`).
```bash
dw=<path-to-dw-4D-data>
dw_corr=<path-to-corrected-dw-4D-data>
bval=<path-to-bval-file>
bvec=<path-to-bvec-file>
bvec_rot=<path-to-rotated-bvec-file>
model_dir=<path-to-models>

python ${eddeep_dir}/scripts/apply_correction.py -i ${dw}\
                                                 -o ${dw_corr}\
                                                 -tr ${model_dir}/trans_gen_best.keras\
                                                 -reg ${model_dir}/corr_best.keras\
                                                 -b ${bval}\
                                                 -g ${bvec}\
                                                 -og ${bvec_rot}
```
Rotated b-vectors (`-og`): the estimated transformation includes a rigid component $R(x) = Ox + t$, relative to the first b=0 volume, with $O$ a rotation matrix and $t$ a translation, expressed in the voxel coordinates of the isotropic grid on which the transformation is estimated. The corrected volume is obtained by sampling the acquired volume at the full estimated transformation, composed of $R$ and an eddy-current component acting only along the phase-encoding direction. The acquired volume therefore shows the subject rotated by $O$ with respect to the reference. A gradient direction $g$ applied during acquisition corresponds to $O^\top g$ relative to the subject in the reference frame, so each b-vector is replaced by $g' = O^\top g$. The translation $t$ does not affect directions and the eddy-current component is not used. Since b-vectors are stored in FSL convention (voxel axes, with the first axis flipped when the orientation matrix has a positive determinant), the rotation is applied in that frame: $g' = F O^\top F g$, with $F = \mathrm{diag}(-1,1,1)$ in the flipped case and $F = I$ otherwise.

## References

If you used **Eddeep** for your work, please cite the following:

&nbsp;[1] A. Legouhy, R. Callaghan, W. Stee, P. Peigneux, H. Azadbakht and H. Zhang.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Eddeep: Fast eddy-current distortion correction for diffusion MRI with deep learning.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MICCAI (2024) [[arxiv]](https://arxiv.org/pdf/2405.10723)

The code uses bits from Neurite and Voxelmorph:

&nbsp;[2] **Voxelmorph** [[github]](https://github.com/voxelmorph/voxelmorph) [[arxiv]](https://arxiv.org/abs/1809.05231)\
&nbsp;[3] **Neurite** [[github]](https://github.com/adalca/neurite)



