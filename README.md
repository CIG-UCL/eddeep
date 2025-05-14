# Eddeep

**Eddeep** is composed of 2 models in sequence:
  1) **Translator**: Restore correspondences between images.
  2) **Registrator**: Estimate the distortion and apply correction.

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
The training and validation data for both models must be organised as follow:
```
├── sub1
│   ├── ped1
│   │   ├── bval1
│   │   │   ├── sub1_ped1_bval1_dir1.nii.gz
│   │   │   ├── sub1_ped1_bval1_dir1.nii.gz
│   │   │   ├── ...
│   │   ├── bval2
│   │   │   ├── ...
│   │   ├── ...
│   │   ├── sub1_ped1_bvaltarget_meandir.nii.gz (only for translation)
│   │   ├── ...
│   └── ped2
│       ├── ...
├── sub2
│   ├── ...
├── ...
```
  - For the translator, the input data is pre-corrected and there is a translation target.
  - For the registrator, the input data is the raw DW data.

### Training the translator
```bash
eddeep_dir=<path-to-eddeep>
out_dir=<path-to-output-dir>
```

```bash
bvaltarget=<chose-target-bvalue>
data_precorr_train_dir=<path-to-precorrected-training-data-dir>
data_precorr_val_dir=<path-to-precorrected-validation-data-dir>

python ${eddeep_dir}/scripts/train_eddeep_trans.py -t ${data_precorr_train_dir}\
                                                   -v ${data_precorr_val_dir}\
                                                   -o ${out_dir}/trans\
                                                   -B ${bvaltarget} -e 1000 -as 0.5 -ai 0.5
```

### Training the registrator
```bash
data_train_dir=<path-to-training-data-dir>
data_val_dir=<path-to-validation-data-dir>

python ${eddeep_dir}/scripts/train_eddeep_corr.py -t ${data_train_dir}\
                                                  -v ${data_val_dir}\
                                                  -tr ${out_dir}/trans_gen_best.h5\
                                                  -o ${out_dir}/corr\
                                                  -p 1\
                                                  -e 200 -as 0.5 -ai 0.5
```
