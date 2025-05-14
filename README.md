# Eddeep

**Eddeep** is composed of 2 models in sequence:
  1) **Translator**: Restore correspondences between images.
  2) **Registrator**: Estimate the distortion and apply correction.

     
## Training Eddeep

### Preprocessing

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
│   │   ├── sub1_ped1_bvaltarget_meandir.nii.gz 
│   │   ├── ...
│   └── ped2
│       ├── ...
├── sub2
│   ├── ...
├── ...
```

#### 1) Pre-correction with an external tool (for translator training only)
During training (but not at inference), the translator takes as input images that have been corrected for eddy distortions by an external tool. You can typically use FSL Eddy or Tortoisefor that.

#### 2) Creation of the translation targets
  - Choose a moderately high (700-3000) b-value among the acquired ones.\
  - For each subject, average all the volumes for this b-value to obtain a direction average image (assuming b-vectors are uniformly sampled on the sphere).



```bash
eddeep_dir=<path-to-eddeep>
data_precorr_train_dir=<path-to-precorrected-training-data-dir>
data_precorr_val_dir=<path-to-precorrected-validation-data-dir>
data_train_dir=<path-to-training-data-dir>
data_val_dir=<path-to-validation-data-dir>
out_dir=<path-to-output-dir>
```

### Training the translator
```bash
python ${eddeep_dir}/scripts/train_eddeep_trans.py -t ${data_precorr_train_dir}\
                                                   -v ${data_precorr_val_dir}\
                                                   -o ${out_dir}/trans\
                                                   -B 2000 -e 50 -as 0.5 -ai 0.5
```

### Training the registrator
```bash
python ${eddeep_dir}/scripts/train_eddeep_corr.py -t ${data_train_dir}\
                                                  -v ${data_val_dir}\
                                                  -tr ${out_dir}/trans_gen_best.h5\
                                                  -o ${out_dir}/corr\
                                                  -p 1\
                                                  -e 50 -as 0.5 -ai 0.5
```
