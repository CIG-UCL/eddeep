import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import scipy
import eddeep.utils



def interp_dw(b0, dw, eps=1e-9):
    
    b0_log = sitk.Log(b0 + eps)
    dw_log = sitk.Log(dw + eps)
        
    alpha = float(2.02-scipy.stats.skewnorm.rvs(5,1,0.05,1)[0])
    b0 = sitk.Exp(alpha*b0_log + (1-alpha)*dw_log) - eps
    
    beta = float(2.2-scipy.stats.skewnorm.rvs(5,1,0.5,1)[0])
    dw = sitk.Exp(beta*dw_log + (1-beta)*b0_log) - eps
    
    return b0, dw


class intensity_aug:
     
    def __init__(self, img_geom):
        
        self.img_geom = img_geom
        self.ndims = img_geom.GetDimension()
        self.dtype = img_geom.GetPixelID()
        self.noise_params = None
        self.bias_field_params = None
        
    def transform(self, img_list):
        
        img_aug_list = []
        for img in img_list:
            
            img_aug = img
            
            if self.bias_field_params is not None:
                bias_field = self.gen_bias_field(self.bias_field_params)
                bias_field = sitk.Cast(bias_field, self.dtype)
                img_aug *= 1 + bias_field
                
            if self.noise_params is not None:
                noise = self.gen_noise(self.noise_params, img)
                noise = sitk.Cast(noise, self.dtype)
                img_aug += noise
                
            img_aug_list += [img_aug]
            
        return img_aug_list
        
        
    def set_bias_field_params(self, shrink_factor=8, smooth_factor=16, field_std_max=3):
        self.bias_field_params = {'shrink_factor': shrink_factor,
                                  'smooth_factor': smooth_factor,
                                  'field_std_max': field_std_max}
        
    def set_noise_params(self, distrib='rician', max_snr=30):
        self.noise_params = {'distrib': distrib,
                             'max_snr': max_snr}
        
    def gen_bias_field(self, bias_field_params):
        
        shrink_factor = bias_field_params['shrink_factor']
        smooth_factor = bias_field_params['smooth_factor']
        field_std_max = bias_field_params['field_std_max']
        
        shrink_img = sitk.Shrink(self.img_geom, [shrink_factor]*self.ndims)
        shrinked_shape = list(shrink_img.GetSize()[::-1])            
        field_std = 2*field_std_max*(np.random.rand(1)-0.5)
        field = field_std * np.random.normal(size = shrinked_shape)
        field_img = sitk.GetImageFromArray(field)
        field_img.CopyInformation(shrink_img)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.img_geom)
        resampler.SetUseNearestNeighborExtrapolator(True)
        field_img = resampler.Execute(field_img)
        
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(smooth_factor)
        field_img = gaussian.Execute(field_img)
        
        return field_img

    
    def gen_noise(self, noise_params, x_img):
        
        max_snr = noise_params['max_snr']
        distrib = noise_params['distrib']
        
        x = sitk.GetArrayFromImage(x_img)
        snr = max_snr * np.random.rand()
        sample_signal = np.quantile(x[x>0].ravel(), 0.8)
        sigma = sample_signal / snr
        
        if distrib in ('rician', 'gaussian'):
            noise = np.random.normal(0, sigma, x.shape)
        
            if distrib == 'rician':
                noise_imag = np.random.normal(0, sigma, x.shape)
                x_real_noisy = x + noise
                x_imag_noisy = noise_imag
                noise = np.sqrt(x_real_noisy**2 + x_imag_noisy**2) - x
        
        else:
            noise = np.zeros_like(x)
            
        noise_img = sitk.GetImageFromArray(noise)
        noise_img.CopyInformation(self.img_geom)
        
        return noise_img
        

def skew_symmetric(v):
    ndims = len(v)

    if ndims == 2:
        return np.array([[0, -v[1]],
                         [v[1], 0]])
    elif ndims == 3:
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])       
    
class spatial_aug:
    
    def __init__(self, img_geom, dire=None, interp=sitk.sitkLinear):
        self.img_geom = img_geom
        self.ndims = img_geom.GetDimension()
        self.spacing = img_geom.GetSpacing()
        self.size = img_geom.GetSize()
        center_vox = [int(float(self.size[k]) / 2) for k in range(self.ndims)]
        self.center = img_geom.TransformIndexToPhysicalPoint(center_vox)
        self.dire = dire
        self.aff_params = None
        self.diffeo_params = None
        self.polyaff_params = None
        self.transfo = None
        self.volshape = np.flip(self.size)
        self.interp = interp
        
    def transform(self, img_list):
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.img_geom)
        resampler.SetTransform(self.transfo)  
        resampler.SetInterpolator(self.interp)
        img_aug_list = []
        for img in img_list:
            img_aug_list += [resampler.Execute(img)]
        
        return img_aug_list
    
    
    def gen_transfo(self):
        
        transfo = sitk.CompositeTransform(self.ndims)
        if self.aff_params is not None:
            self.affine_transfo = self.gen_aff_transfo(self.aff_params)
            transfo.AddTransform(self.affine_transfo)
        if self.diffeo_params is not None:
            self.diffeo_transfo = self.gen_diffeo_transfo(self.diffeo_params)
            transfo.AddTransform(self.diffeo_transfo)
        if self.polyaff_params is not None:
            if self.polyaff_params['gpu']:
                self.polyaff_transfo = self.gen_polyaff_transfo_gpu(self.polyaff_params)
            else:
                self.polyaff_transfo = self.gen_polyaff_transfo(self.polyaff_params)
            transfo.AddTransform(self.polyaff_transfo)
        self.transfo = transfo
    
    def get_aff_transfo(self):
        return self.affine_transfo
    
    def get_polyaff_transfo(self):
        return self.affine_transfo
    
    def get_diffeo_transfo(self):
        return self.diffeo_transfo
        
    def get_transfo(self):
        return self.transfo
    
    def set_transfo(self, transfo):
        self.transfo = transfo
    
        
    def set_aff_params(self, trans_bounds=5, rot_bounds=np.pi/8, scalDir_bounds=np.pi/12, scal_bounds=1.3):
        self.aff_params = {'trans_bounds': trans_bounds,
                           'rot_bounds': rot_bounds,
                           'scalDir_bounds': scalDir_bounds,
                           'scal_bounds': scal_bounds}

    def set_polyaff_params(self, nb_points=1000, sigma=15, trans_bounds=5, rot_bounds=np.pi/2, scalDir_bounds=np.pi/4, scal_bounds=5, int_steps=6, gpu=False, disp=False):
        self.polyaff_params = {'nb_points': nb_points,
                               'sigma': sigma,
                               'trans_bounds': trans_bounds,
                               'rot_bounds': rot_bounds,
                               'scalDir_bounds': scalDir_bounds,
                               'scal_bounds': scal_bounds,
                               'int_steps': int_steps,
                               'gpu': gpu}
        
    def set_diffeo_params(self, shrink_factor=8, smooth_factor=6, svf_std_max=8, int_steps=4):
        self.diffeo_params = {'shrink_factor': shrink_factor,
                              'smooth_factor': smooth_factor,
                              'svf_std_max': svf_std_max,
                              'int_steps': int_steps}
        
    def gen_aff_transfo(self, aff_params, center=None, format='itk'):
        
        trans_bounds = aff_params['trans_bounds']
        rot_bounds = aff_params['rot_bounds']
        scalDir_bounds = aff_params['scalDir_bounds']
        scal_bounds = aff_params['scal_bounds']
        if center is None:
            center = np.array(self.center)

        theta = 2*rot_bounds*(np.random.rand()-0.5)
        theta_scal = 2*scalDir_bounds*(np.random.rand()-0.5)
        
        if self.ndims == 2:
            rot = skew_symmetric([1])
            scalDir = skew_symmetric([1])
            
        elif self.ndims == 3:
            rotax = np.random.randn(3)
            rotax /= np.linalg.norm(rotax)
            rot = skew_symmetric(rotax)     
            rotax_scal = np.random.randn(3)
            rotax_scal /= np.linalg.norm(rotax_scal)
            scalDir = skew_symmetric(rotax_scal)

        rot = scipy.linalg.expm(theta * rot)
        scalDir = scipy.linalg.expm(theta_scal * scalDir)
        
        scal_factors = np.exp(2*np.log(scal_bounds)*(np.random.rand(self.ndims)-0.5))
        scal = np.diag(scal_factors)
        
        trans = 2*trans_bounds*(np.random.rand(self.ndims)-0.5)
        
        affine = np.matmul(rot,np.matmul(scalDir,np.matmul(scal,np.transpose(scalDir))))
        trans = np.matmul(affine, -center) + trans + center

        if format == 'itk':
            affine_transfo = sitk.AffineTransform(self.ndims)
            affine_transfo.SetMatrix(np.ravel(affine)) 
            affine_transfo.SetTranslation(trans)
        else:
            affine_transfo = np.c_[affine, trans[...,None]]
            affine_transfo = np.r_[affine_transfo, np.array([[0]*self.ndims + [1]])]

        return affine_transfo
    

    def gen_polyaff_transfo(self, polyaff_params):
        
        nb_points = polyaff_params['nb_points']
        sigma = polyaff_params['sigma']
        int_steps = polyaff_params['int_steps']
        
        theta = (sigma**2 + 4*(sigma/2)**2)**(1/2)/2 - sigma/2
        k = (sigma*(sigma + (sigma**2 + 4*(sigma/2)**2)**(1/2)))/(2*(sigma/2)**2) + 1

        id2 = sitk.AffineTransform(self.ndims)
        id2.SetMatrix(2*np.eye(self.ndims).ravel())
        trsf2disp = sitk.TransformToDisplacementFieldFilter()
        trsf2disp.SetReferenceImage(self.img_geom)
        polyaff_svf = sitk.Image(self.size, sitk.sitkVectorFloat64)
        polyaff_svf.CopyInformation(self.img_geom)
        weight_map_sum = sitk.Image(self.size, sitk.sitkFloat64)
        weight_map_sum.CopyInformation(self.img_geom) 
        loc_transfo = sitk.AffineTransform(self.ndims)
        
        upper_vox = np.array(self.size)
        lower_vox = np.array([0]*self.ndims)
        center_vox = (upper_vox + lower_vox) / 2 
        upper_vox = 1.1 * (upper_vox - center_vox) + center_vox
        lower_vox = 1.1 * (lower_vox - center_vox) + center_vox
        
        r = scipy.stats.qmc.Sobol(self.ndims).random(nb_points) 
        
        for i in range(nb_points):
            # print(i)
            
            # Get (pseudo) random control point 
            point_vox = r[i,:]*upper_vox + (1-r[i,:])*lower_vox 
            point = np.array(self.img_geom.TransformContinuousIndexToPhysicalPoint(point_vox))
            
            # Get random affine transfo
            affine_transfo = self.gen_aff_transfo(polyaff_params, center=point)  
            loc_mat = np.concatenate((np.reshape(affine_transfo.GetMatrix(), [self.ndims]*2),
                                      np.reshape(affine_transfo.GetTranslation(), [self.ndims,1])), axis=1)
            loc_mat = np.concatenate((loc_mat, [[0]*self.ndims + [1]]), axis=0)
            loc_mat = scipy.linalg.logm(loc_mat)
            if not np.isrealobj(loc_mat):
                continue
            
            # Compute weight map
            sigma_i = np.random.gamma(k, theta)
            id2.SetTranslation(-point) 
            weight_map = trsf2disp.Execute(id2)    
            weight_map = sitk.VectorMagnitude(weight_map)**2          
            weight_map = sitk.Exp(-weight_map/(2*sigma_i**2)) 
            
            # Update polyaffine with current field
            loc_transfo.SetMatrix((loc_mat[0:self.ndims, 0:self.ndims] + np.eye(self.ndims)).ravel())
            loc_transfo.SetTranslation(loc_mat[0:self.ndims, self.ndims])
            loc_svf = trsf2disp.Execute(loc_transfo)
            polyaff_svf += sitk.Compose([sitk.VectorIndexSelectionCast(loc_svf,d)*weight_map for d in range(self.ndims)])
            weight_map_sum += weight_map

        polyaff_svf = sitk.Compose([sitk.VectorIndexSelectionCast(polyaff_svf, d)/weight_map_sum for d in range(self.ndims)])
        polyaff = eddeep.utils.integrate_svf(polyaff_svf, int_steps=int_steps)
           
        return polyaff
    
 
    def gen_polyaff_transfo_gpu(self, polyaff_params):

        nb_points = polyaff_params['nb_points']
        sigma = polyaff_params['sigma']
        int_steps = polyaff_params['int_steps']
        
        theta = (sigma**2 + 4*(sigma/2)**2)**(1/2)/2 - sigma/2
        k = (sigma*(sigma + (sigma**2 + 4*(sigma/2)**2)**(1/2)))/(2*(sigma/2)**2) + 1

        polyaff_svf = tf.zeros([self.ndims, np.prod(self.volshape)])
        weight_map_sum = tf.zeros(np.prod(self.volshape))
 
        matO = tf.cast(eddeep.utils.get_matOrientation(self.img_geom), tf.float32)
        grid = tf.meshgrid(*[tf.range(sz) for sz in self.volshape], indexing='ij')
        grid = tf.cast(grid, dtype=tf.float32) 
        grid = tf.stack([tf.reshape(grid[d], -1) for d in range(self.ndims)], axis=0)       
        grid = tf.concat((grid, tf.ones((1,tf.reduce_prod(self.volshape)), dtype=tf.float32)), axis=0)
        grid = tf.matmul(matO, grid)
  
        upper_vox = tf.constant(self.volshape, dtype=tf.float32)
        lower_vox = tf.zeros(self.ndims, dtype=tf.float32)
        center_vox = (upper_vox + lower_vox) / 2 
        upper_vox = tf.reshape(1.1 * (upper_vox - center_vox) + center_vox, (self.ndims,1))
        lower_vox = tf.reshape(1.1 * (lower_vox - center_vox) + center_vox, (self.ndims,1))
        
        # Get (pseudo) random control point  
        r = scipy.stats.qmc.Sobol(self.ndims).random(nb_points).T
        r = tf.constant(r, dtype=tf.float32)       
        points = r*upper_vox + (1-r)*lower_vox
        points = tf.concat((points, tf.ones((1, nb_points))), axis=0)
        points = tf.matmul(matO, points)[:self.ndims,:] 

        # Get random affine transfo
        aff_mats = [self.gen_aff_transfo(polyaff_params, center=points[:,i].numpy(), format='mat') for i in range(nb_points)]
        aff_mats = tf.stack(aff_mats, axis=0)  
        aff_mats = tf.cast(aff_mats, dtype=tf.complex64)
        aff_mats = tf.math.real(tf.linalg.logm(aff_mats))

        for i in range(nb_points):
            loc_mat = aff_mats[i, ...]
            point = tf.reshape(points[:,i], (self.ndims,1))
            
            # Compute weight map
            sigma_i = np.random.gamma(k, theta)  # sigma_i = tf.random.gamma([1], k, theta)
            weight_map = tf.reduce_sum((grid[:self.ndims,:] - point)**2, axis=0)
            weight_map = tf.exp(-weight_map/(2*sigma_i**2))            
            
            # Update polyaffine with current field
            polyaff_svf += tf.matmul(loc_mat, grid)[:self.ndims,:] * weight_map        
            weight_map_sum += weight_map

        weight_map_sum += 1e-5*sigma*tf.sqrt(2*np.pi)
        polyaff_svf /= weight_map_sum
        
        weight_map_sum = tf.reshape(weight_map_sum, self.volshape)
        polyaff_svf = np.stack([np.reshape(polyaff_svf[d, ...], self.volshape)
                                for d in range(self.ndims)], axis=-1)
        
        polyaff = sitk.GetImageFromArray(polyaff_svf)
        polyaff.CopyInformation(self.img_geom)
        polyaff = eddeep.utils.integrate_svf(polyaff, int_steps=int_steps)
 
        return polyaff
    
    
    def gen_diffeo_transfo(self, diffeo_params):

        shrink_factor = diffeo_params['shrink_factor']
        smooth_factor = diffeo_params['smooth_factor']
        svf_std_max = diffeo_params['svf_std_max']
        int_steps = diffeo_params['int_steps']
        
        shrink_img = sitk.Shrink(self.img_geom, [shrink_factor]*self.ndims)
        shrinked_shape = list(shrink_img.GetSize()[::-1])            
        svf_std = 2*svf_std_max*(np.random.rand(1)-0.5)
        svf = svf_std * np.random.normal(size = shrinked_shape + [self.ndims])
        svf_img = sitk.GetImageFromArray(svf, isVector=True)
        svf_img.CopyInformation(shrink_img)
        
        shrink_img = sitk.Shrink(self.img_geom, [2]*self.ndims)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(shrink_img)
        resampler.SetUseNearestNeighborExtrapolator(True)
        svf_img = resampler.Execute(svf_img)
        
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(smooth_factor)
        svf_img = gaussian.Execute(svf_img)
        svf_img = sitk.Cast(svf_img, sitk.sitkVectorFloat64)
        
        transfo = eddeep.utils.integrate_svf(svf_img, int_steps=int_steps)
        
        return sitk.DisplacementFieldTransform(transfo)
        


class spatial_aug_dir:
    """
    Keep lines parallel to dire that way after transfo.
    """
    def __init__(self, img_geom, dire=1):
        img_geom.SetDirection([1,0,0,0,1,0,0,0,1])
        img_geom.SetOrigin([0,0,0])
        img_geom.SetSpacing([2,2,2])
        self.img_geom = img_geom
        self.ndims = img_geom.GetDimension()
        self.spacing = img_geom.GetSpacing()
        self.size = img_geom.GetSize()
        center_vox = [int(float(self.size[k]) / 2) for k in range(self.ndims)]
        self.center = img_geom.TransformIndexToPhysicalPoint(center_vox)
        self.dire = dire
        self.aff_params = None
        self.diffeo_params = None
        
    def transform(self, img_list):
 
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.img_geom)
        resampler.SetTransform(self.transfo)
        img_aug_list = []
        for img in img_list:
            img.SetDirection([1,0,0,0,1,0,0,0,1])
            img.SetOrigin([0,0,0])
            img.SetSpacing([2,2,2])
            img_aug_list += [resampler.Execute(img)]
        
        return img_aug_list
    
    
    def gen_transfo(self):
        
        transfo = sitk.CompositeTransform(self.ndims)
        if self.aff_params is not None:
            self.affine_transfo = self.gen_aff_transfo(self.aff_params)
            transfo.AddTransform(self.affine_transfo)
        if self.diffeo_params is not None:
            self.diffeo_transfo = self.gen_diffeo_transfo(self.diffeo_params)
            transfo.AddTransform(self.diffeo_transfo)
        self.transfo = transfo
    
    
    def set_aff_params(self, trans_bounds=5, rot_bounds=np.pi/8, shear_bounds=0.3, scal_bounds=1.3):
        self.aff_params = {'trans_bounds': trans_bounds,
                           'rot_bounds': rot_bounds,
                           'shear_bounds': shear_bounds,
                           'scal_bounds': scal_bounds}
        
    def set_diffeo_params(self, shrink_factor=8, smooth_factor=6, svf_std_max=8, int_steps=4):
        self.diffeo_params = {'shrink_factor': shrink_factor,
                              'smooth_factor': smooth_factor,
                              'svf_std_max': svf_std_max,
                              'int_steps': int_steps}
        
    def gen_aff_transfo(self, aff_params, center=None, format='itk'):
        
        trans_bounds = self.aff_params['trans_bounds']
        rot_bounds = self.aff_params['rot_bounds']
        shear_bounds = self.aff_params['shear_bounds']
        scal_bounds = self.aff_params['scal_bounds']
        if center is None:
            center = np.array(self.center)
            
        theta = 2*rot_bounds*(np.random.rand()-0.5)
        rotax = np.zeros(3)
        rotax[self.dire] = 1
        rot = skew_symmetric(rotax)
        rot = scipy.linalg.expm(theta * rot)
        
        shear = np.eye(3)
        shear_factors = 2*shear_bounds*(np.random.rand(2)-0.5)
        if self.dire == 0:
            shear[0,1] = shear_factors[0]
            shear[0,2] = shear_factors[1]
        elif self.dire == 1:
            shear[1,0] = shear_factors[0]
            shear[1,2] = shear_factors[1]
        elif self.dire == 2:
            shear[2,0] = shear_factors[0]
            shear[2,1] = shear_factors[1]
        
        scal_factors = np.exp(2*np.log(scal_bounds)*(np.random.rand(3)-0.5))
        scal = np.diag(scal_factors)
        
        trans = 2*trans_bounds*(np.random.rand(3)-0.5)
        
        affine = np.matmul(rot,np.matmul(scal,shear))
        trans = np.matmul(affine, -center) + trans + center
        
        if format == 'itk':
            affine_transfo = sitk.AffineTransform(self.ndims)
            affine_transfo.SetMatrix(np.ravel(affine)) 
            affine_transfo.SetTranslation(trans)
        else:
            affine_transfo = np.c_[affine, trans[...,None]]
            affine_transfo = np.r_[affine_transfo, np.array([[0]*self.ndims + [1]])]

        return affine_transfo


    def gen_diffeo_transfo(self, diffeo_params):

        shrink_factor = self.diffeo_params['shrink_factor']
        smooth_factor = self.diffeo_params['smooth_factor']
        svf_std_max = self.diffeo_params['svf_std_max']
        int_steps = self.diffeo_params['int_steps']
        
        shrink_img = sitk.Shrink(self.img_geom, [shrink_factor]*self.ndims)
        shrinked_shape = list(shrink_img.GetSize()[::-1])            
        svf_std = 2*svf_std_max*(np.random.rand(1)-0.5)
        svf = np.zeros(shrinked_shape + [self.ndims])
        svf[..., self.dire] = svf_std * np.random.normal(size = shrinked_shape)
        svf_img = sitk.GetImageFromArray(svf, isVector=True)
        svf_img.CopyInformation(shrink_img)
        
        shrink_img = sitk.Shrink(self.img_geom, [2]*self.ndims)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(shrink_img)
        resampler.SetUseNearestNeighborExtrapolator(True)
        svf_img = resampler.Execute(svf_img)
        
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(smooth_factor)
        svf_img = gaussian.Execute(svf_img)
        svf_img = sitk.Cast(svf_img, sitk.sitkVectorFloat64)
        
        transfo = eddeep.utils.integrate_svf(svf_img, int_steps=int_steps)

        return sitk.DisplacementFieldTransform(transfo)
        