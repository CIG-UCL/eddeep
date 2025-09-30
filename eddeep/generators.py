import numpy as np
import os
import glob
import SimpleITK as sitk       
 
import eddeep.utils
import eddeep.augmentation


def nb_vols_per_bval(sub, ped):
    
    list_bval = sorted(next(os.walk(os.path.join(sub, ped)))[1])
    list_bval.remove('b0')
    nb_vols = []
    for bval in list_bval:
        list_dw = sorted(glob.glob(os.path.join(sub, ped, bval, '*.nii*')))
        nb_vols += [len(list_dw)]

    return nb_vols, list_bval


def eddeep_fromDWI(subdirs,
                   k = 4,
                   ped = None,
                   bval = None,
                   dw_file = None,
                   b0_file = None,
                   target_bval = 'same',
                   sl_axi = None,
                   get_dwmean = False,
                   spat_aug_prob=0,
                   int_aug_prob=0,
                   aug_dire = None,
                   prob_bval=True,
                   batch_size=1):
    # sub
    #   |_ PED
    #        |_ bval
    
    while True:
        b0s = []
        dws = []
        dws_mean = []
        
        for b in range(batch_size):
            # random subject
            i = np.random.choice(range(0, len(subdirs)))
            sub_i = subdirs[i]
            
            # random PED
            if ped is None:
                list_ped = sorted(next(os.walk(sub_i))[1])
                ind_ped = np.random.choice(range(0, len(list_ped)))
                ped_i = list_ped[ind_ped]
            else:
                ped_i = ped
            
            # random bval
            if bval is None:
                nb_vols, list_bval = nb_vols_per_bval(sub_i, ped_i)
                if prob_bval:
                    list_bval = np.repeat(list_bval, nb_vols).tolist()
                ind_bval = np.random.choice(range(len(list_bval)))
                bval_i = list_bval[ind_bval]
            else: 
                bval_i = 'b' + str(bval)
            if target_bval == 'same':
                target_bval = bval
                
            # random DW and b=0 image  
            if b0_file is None:
                list_b0 = sorted(glob.glob(os.path.join(sub_i, ped_i, 'b0', '*.nii*')))
                list_b0 = [os.path.split(list_b0[j])[-1] for j in range(len(list_b0))]
                ind_b0 = np.random.choice(range(0, len(list_b0)))   
                b0_file_i = list_b0[ind_b0]
            else:
                b0_file_i = b0_file
            
            if dw_file is None:
                list_dw = sorted(glob.glob(os.path.join(sub_i, ped_i, bval_i, '*.nii*')))
                list_dw = [os.path.split(list_dw[j])[-1] for j in range(len(list_dw))]
                ind_dw = np.random.choice(range(0, len(list_dw))) 
                dw_file_i = list_dw[ind_dw]
            else:
                dw_file_i = dw_file
                
            b0 = sitk.ReadImage(os.path.join(sub_i, ped_i, 'b0', b0_file_i))
            b0 = sitk.Cast(b0, sitk.sitkFloat32)
            b0 = sitk.Clamp(b0, lowerBound=0.0)
            b0 = eddeep.utils.pad_image(b0, k=k)
            
            dw = sitk.ReadImage(os.path.join(sub_i, ped_i, bval_i, dw_file_i))
            dw = sitk.Cast(dw, sitk.sitkFloat32)
            dw = sitk.Clamp(dw, lowerBound=0.0)
            dw = eddeep.utils.pad_image(dw, k=k)
            
            if get_dwmean:
                dw_mean = sitk.ReadImage(glob.glob(os.path.join(sub_i, ped_i,'*_b' + str(target_bval) + '_mean.nii.gz'))[0])
                dw_mean = sitk.Cast(dw_mean, sitk.sitkFloat32)
                dw_mean = sitk.Clamp(dw_mean, lowerBound=0.0)
                dw_mean = eddeep.utils.pad_image(dw_mean, k=k)
                
            if np.random.rand() < spat_aug_prob:
                if aug_dire is not None:
                    aug = eddeep.augmentation.spatial_aug_dir(b0, dire=aug_dire)
                else:
                    aug = eddeep.augmentation.spatial_aug(b0)
                aug.set_aff_params()
                aug.set_diffeo_params()  
                aug.gen_transfo()
                if get_dwmean:
                    b0, dw, dw_mean = aug.transform([b0, dw, dw_mean])
                else:
                    b0, dw = aug.transform([b0, dw])
                    
            if np.random.rand() < int_aug_prob:
                
                b0, dw = eddeep.augmentation.interp_dw(b0, dw)
                
                aug_int = eddeep.augmentation.intensity_aug(b0)
                aug_int.set_bias_field_params()
                aug_int.set_noise_params()
                b0 = aug_int.transform([b0])[0]
                
                aug_int = eddeep.augmentation.intensity_aug(dw)
                aug_int.set_bias_field_params()
                aug_int.set_noise_params()
                dw = aug_int.transform([dw])[0]
                
            b0 = sitk.GetArrayFromImage(b0)[np.newaxis,..., np.newaxis]
            b0 = eddeep.utils.normalize_intensities_q(b0, 0.999)
                        
            dw = sitk.GetArrayFromImage(dw)[np.newaxis,..., np.newaxis]
            dw = eddeep.utils.normalize_intensities_q(dw, 0.999)
            
            if get_dwmean:                
                dw_mean = sitk.GetArrayFromImage(dw_mean)[np.newaxis,..., np.newaxis]
                dw_mean = eddeep.utils.normalize_intensities_q(dw_mean, 0.999)
            
            if sl_axi is not None:
                b0 = b0[:,sl_axi,:,:,:]
                dw = dw[:,sl_axi,:,:,:]
                dw_mean = dw_mean[:,sl_axi,:,:,:]
                
            b0s += [b0]
            dws += [dw]
            if get_dwmean:
                dws_mean += [dw_mean]
        
        if get_dwmean:
            yield [np.concatenate(b0s,axis=0), np.concatenate(dws,axis=0), np.concatenate(dws_mean,axis=0)]
        else:
            yield [np.concatenate(b0s,axis=0), np.concatenate(dws,axis=0)]
        


def eddeep_fromDWI_test(dw_file,
                        out_size,
                        dwmean_file = None):

    dws_img = sitk.ReadImage(dw_file)
    n_vol = dws_img.GetSize()[-1]

    dws = []
    for b in range(n_vol):

        dw = sitk.Cast(dws_img[:,:,:,b], sitk.sitkFloat32)
        dw = sitk.Clamp(dw, lowerBound=0.0)
        dw = eddeep.utils.pad_image(dw, out_size=out_size)
        dw = sitk.GetArrayFromImage(dw)[np.newaxis,..., np.newaxis]
        dw = eddeep.utils.normalize_intensities_q(dw, 0.999)
        dws += [dw]
        
    dws = np.concatenate(dws, axis=0)
    
    if dwmean_file is not None:
        dw_mean = sitk.ReadImage(dwmean_file)
        dw_mean = sitk.Cast(dw_mean, sitk.sitkFloat32)
        dw_mean = sitk.Clamp(dw_mean, lowerBound=0.0)
        dw_mean = eddeep.utils.pad_image(dw_mean, out_size=out_size)
        dw_mean = sitk.GetArrayFromImage(dw_mean)[np.newaxis,..., np.newaxis]
        dw_mean = eddeep.utils.normalize_intensities_q(dw_mean, 0.999)

        return dws, dw_mean
    
    else:
        return dws
    