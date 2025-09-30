import os
import copy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd

from external import neurite as ne


def develop(x, dec='|_'):
    
    if isinstance(x, (list, tuple)):
        print(dec, type(x), len(x))
        dec = '|    ' + dec
        for i in range(len(x)):
            develop(x[i], dec)
    elif isinstance(x, np.ndarray) or tf.is_tensor(x):
        print(dec, type(x), x.dtype, x.shape)   
    else: 
        print(dec, type(x))   

    
def shift_to_transfo(loc_shift, indexing='ij'):
    
    if isinstance(loc_shift.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = loc_shift.shape[1:-1].as_list()
    else:
        volshape = loc_shift.shape[1:-1]
    ndims = len(volshape)
    
    ij = [range(volshape[i]) for i in range(ndims)]
    mesh = tf.meshgrid(*ij, indexing=indexing)
    mesh = tf.cast(tf.expand_dims(tf.stack(mesh, axis=-1), 0), 'float32')
    
    return mesh + loc_shift


def plot_losses(loss_file, is_val=False, do_log=False, write=True,
                suptitle='', reord_ind=None, nb_rows=1,
                ymin=[None], ymax=[None], xmin=[None], xmax=[None]):

    tab_loss = pd.read_csv(loss_file, sep=',')
    if reord_ind is not None:
        reord_cols = tab_loss.columns[reord_ind]
        tab_loss = tab_loss[reord_cols]

    nb_losses = len(tab_loss.columns) - 1
    if is_val:
        nb_losses = int(nb_losses / 2)
    nb_cols = int(np.ceil(nb_losses / nb_rows))
    
    if len(xmin) == 1: xmin = xmin * nb_losses
    if len(xmax) == 1: xmax = xmax * nb_losses
    if len(ymin) == 1: ymin = ymin * nb_losses
    if len(ymax) == 1: ymax = ymax * nb_losses
    
    f, axs = plt.subplots(nb_rows, nb_cols, figsize=(12,5))
    if nb_rows > 1:
        axs = np.ravel(axs)
    if nb_losses == 1: axs = [axs]
    f.dpi = 200
    plt.rcParams['font.size'] = '10'
    plt.rcParams["xtick.major.size"] = 2
    plt.rcParams["ytick.major.size"] = 2
     
    for l in range(nb_losses):
        if is_val:
            if do_log:
                yloss = np.log(tab_loss.loc[:,tab_loss.columns[nb_losses+l+1]])
            else:
                yloss = tab_loss.loc[:,tab_loss.columns[nb_losses+l+1]]
            axs[l].plot(tab_loss.epoch, yloss, label="validation")
            
        if do_log:
            yloss = np.log(tab_loss.loc[:,tab_loss.columns[l+1]])
        else:
            yloss = tab_loss.loc[:,tab_loss.columns[l+1]]
        axs[l].plot(tab_loss.epoch, yloss, label="training")
        
        
        axs[l].set_title(tab_loss.columns[l+1], fontsize=12)
        axs[l].legend(prop={'size': 10})

        if xmin[l] is not None: axs[l].set_xlim(left=xmin[l])
        if xmax[l] is not None: axs[l].set_xlim(right=xmax[l])
        if ymin[l] is not None: axs[l].set_ylim(bottom=ymin[l])    
        if ymax[l] is not None: axs[l].set_ylim(top=ymax[l])
        axs[l].grid(axis='y')
    plt.tight_layout()
    plt.suptitle(suptitle)
    
    if write:
        filename, _ = os.path.splitext(loss_file)
        plt.savefig(filename + '.png')


def get_matOrientation(img, indexing='itk'):
    # CAREFUL: 
    # It's from itk indices to physical space by default.
    # For numpy indices to physical space, use indexing='numpy'.

    ndims = img.GetDimension() 
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    
    perm = np.eye(ndims+1)
    if indexing == 'numpy':
        perm[:ndims,:ndims] = np.eye(ndims)[::-1]
    
    matO = np.matmul(np.reshape(direction,(ndims, ndims)), np.diag(spacing))
    matO = np.concatenate((matO, np.reshape(origin, (ndims,1))), axis=1)
    matO = np.concatenate((matO, np.reshape([0]*ndims+[1], (1,ndims+1))), axis=0)
    matO = np.matmul(matO, perm)

    return matO


def decomp_matOrientation(matO):
    """
    Decompose the orientation matrix into origin, scaling and direction.
    """
    
    ndims = matO.shape[1]-1
    mat = matO[0:ndims, 0:ndims]   
    spacing = np.linalg.norm(mat, axis=0)
    direction = np.squeeze(np.asarray(np.matmul(mat, np.diag(1/spacing))))
    origin = np.squeeze(np.asarray(matO[0:ndims, ndims]))
    
    return (origin, spacing, direction.ravel())



def resample_image(img, size, matO, interp):
    """
    Resample an image in a new grid defines by its size and orientation using a given interpolation method.
    """
    
    origin, spacing, direction = decomp_matOrientation(matO)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputOrigin(origin.tolist())
    resampler.SetOutputDirection(direction.tolist())
    resampler.SetOutputSpacing(spacing.tolist())   
    resampler.SetInterpolator(interp)
    
    return resampler.Execute(img)


def normalize_intensities(img, wmin=0, wmax=None, omin=0, omax=1, dtype=sitk.sitkFloat32):
    """
    Normalize intensities of an itk image between 0 and 1.
    """
    listed = True
    if not isinstance(img, (list, tuple)):
        img = [img]
        listed = False
    
    intensityFilter = sitk.IntensityWindowingImageFilter() 
    intensityFilter.SetOutputMaximum(omax)
    intensityFilter.SetOutputMinimum(omin)
    intensityFilter.SetWindowMinimum(wmin)
        
    for i in range(len(img)):
        if dtype is not None:
            img[i] = sitk.Cast(img[i], dtype)
    
    if wmax is None:
        minmaxFilter = sitk.MinimumMaximumImageFilter()
        wmax = -np.inf
        for i in range(len(img)):
            minmaxFilter.Execute(img[i])
            wmax = np.max((wmax, minmaxFilter.GetMaximum()))
    intensityFilter.SetWindowMaximum(wmax)
    
    for i in range(len(img)):
        img[i] = intensityFilter.Execute(img[i])
        
    if not listed:
        img = img[0]
    
    return img
    

def normalize_intensities_q(arr, q=0.99):
    """
    Normalize values of an array between 0 and 1.
    """
    
    val_q = np.quantile(arr, q)
    arr = np.clip(arr, 0, val_q) / val_q
    
    return arr
    


def change_img_res(img, vox_sz=[2,2,2], interp=sitk.sitkLinear):
    """
    Change the resolution while keeping the position and all.
    """
    
    ndims = img.GetDimension()
    direction = list(img.GetDirection()) 
    spacing = list(img.GetSpacing())
    origin = list(img.GetOrigin())
    size = list(img.GetSize())

    size_new = [int(size[d] * spacing[d] / vox_sz[d]) for d in range(ndims)]
    true_vox_sz = [size[d] * spacing[d] / size_new[d] for d in range(ndims)]
    origin_new = [origin[d] + (true_vox_sz[d] - spacing[d]) / 2 for d in range(ndims)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size_new)
    resampler.SetOutputOrigin(origin_new)
    resampler.SetOutputSpacing(true_vox_sz)
    resampler.SetOutputDirection(direction)
    resampler.SetInterpolator(interp)
    
    return resampler.Execute(img)
    

def change_img_size(img, grid_sz=[96,128,96]):
    
    size = img.GetSize()
    
    center = np.flip(np.floor(np.mean(np.array(np.where(sitk.GetArrayFromImage(img))), axis=1)))
    half_sz = np.floor(np.array(grid_sz) / 2)
    bound_inf = (center - half_sz).astype(np.int16)
    bound_sup = (center + grid_sz - half_sz - size).astype(np.int16)
    
    pad_bound_inf = (np.abs(bound_inf) * (bound_inf < 0)).tolist()
    pad_bound_sup = (bound_sup * (bound_sup > 0)).tolist()
    crop_bound_inf = (bound_inf * (bound_inf > 0)).tolist()
    crop_bound_sup = (np.abs(bound_sup) * (bound_sup < 0)).tolist()
    
    img = sitk.ConstantPad(img, pad_bound_inf, pad_bound_sup)
    img = sitk.Crop(img, crop_bound_inf, crop_bound_sup)
    
    return img
              
  
def pad_image(img, k=5, out_size=None, bg_val=0):
    """
    Pad an image such that image size along each dimension is a multiple of 2^k.
    """
    in_size = np.array(img.GetSize(), dtype=np.float32)
    if out_size is None:
        if k is None:
            out_size = np.power(2, np.ceil(np.log(in_size)/np.log(2)))
        else:
            out_size = np.ceil(in_size / 2**k) * 2**k
            
    lowerPad = np.round((out_size - in_size) / 2)
    upperPad = out_size - in_size - lowerPad
    
    padder = sitk.ConstantPadImageFilter()
    padder.SetConstant(bg_val)
    padder.SetPadLowerBound(lowerPad.astype(int).tolist())
    padder.SetPadUpperBound(upperPad.astype(int).tolist())
    
    paddedImg = padder.Execute(img)
    
    return paddedImg


def unpad_image(padded_img, original_size):

    original_size = np.array(original_size, dtype=np.float32)
    padded_size = np.array(padded_img.GetSize(), dtype=np.float32)

    lowerPad = np.round((padded_size - original_size) / 2)

    region_extractor = sitk.RegionOfInterestImageFilter()
    region_extractor.SetSize(original_size.astype(int).tolist())
    region_extractor.SetIndex([int(x) for x in lowerPad])

    return region_extractor.Execute(padded_img)


def jacobian(transfo, outDet=False, dire=None, is_shift=False):
    # takes a tensor of shape [batch_size, sx, sy, (sz,) ndims] as input.
    
    if isinstance(transfo.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = transfo.shape[1:-1].as_list()
    else:
        volshape = transfo.shape[1:-1]
    ndims = len(volshape)
    ndirs = transfo.shape[-1]
    if dire is not None and ndirs != 1:
        raise Exception('the last dim should be of size 1 for a unidirectional field, but got: %s' % ndims)
    
    jacob = []
    for d in range(ndims):
        grad = tf.gather(transfo, range(2, volshape[d]), axis=d+1)-tf.gather(transfo, range(volshape[d]-2), axis=d+1)
        grad_left = tf.gather(transfo, [1], axis=d+1)-tf.gather(transfo, [0], axis=d+1)
        grad_right = tf.gather(transfo, [volshape[d]-1], axis=d+1)-tf.gather(transfo, [volshape[d]-2], axis=d+1)
        grad = tf.concat((grad_left, grad/2, grad_right), axis=d+1)  
        grad = tf.expand_dims(grad, axis=-1)
        jacob += [grad]
    
    jacob = tf.concat(jacob, axis=-1) 
    
    if is_shift:
        if dire is None:
            jacob += tf.eye(ndims, ndims, tf.shape(transfo)[:-1])
        else:
            identity = [tf.ones(tf.shape(transfo)[:-1]) if d==dire else tf.zeros(tf.shape(transfo)[:-1]) for d in range(ndims)]
            jacob += tf.expand_dims(tf.stack(identity, axis=-1), axis=-2)
        
    if outDet:
        # detjac = tf.linalg.det(jacob)
        if ndims == 1:
            detjac = jacob[:,:,0,0]
        elif ndims == 2:
            if ndirs == 2:
                detjac =  jacob[:,:,:,0,0] * jacob[:,:,:,1,1]\
                        - jacob[:,:,:,1,0] * jacob[:,:,:,0,1] 
            elif ndirs == 1:
                detjac = jacob[:,:,:,0,dire]
        elif ndims == 3:
            if ndirs == 3:
                detjac =  jacob[:,:,:,:,0,0] * jacob[:,:,:,:,1,1] * jacob[:,:,:,:,2,2]\
                        + jacob[:,:,:,:,0,1] * jacob[:,:,:,:,1,2] * jacob[:,:,:,:,2,0]\
                        + jacob[:,:,:,:,0,2] * jacob[:,:,:,:,1,0] * jacob[:,:,:,:,2,1]\
                        - jacob[:,:,:,:,2,0] * jacob[:,:,:,:,1,1] * jacob[:,:,:,:,0,2]\
                        - jacob[:,:,:,:,1,0] * jacob[:,:,:,:,0,1] * jacob[:,:,:,:,2,2]\
                        - jacob[:,:,:,:,0,0] * jacob[:,:,:,:,2,1] * jacob[:,:,:,:,1,2]
            elif ndirs == 1:
                detjac = jacob[:,:,:,:,0,dire]
        else:
            raise Exception('Only dimension 2 or 3 supported, but got: %s' % ndims)
            
        return jacob, detjac
    else: 
        return jacob 


class get_real_transfo_aff:
    """
    Compute the real coordinates affine transformation based on
        - A voxelic transformation.
        - An initial real transformation.
        - An orientation matrix.
    This transformation can be used in softwares that properly handle orientation 
    (like ITK-based one, NOT like fsl.)
    """

    def __init__(self, ndims, **kwargs):
        self.ndims = ndims
        if ndims != 3 and ndims != 2:
            raise NotImplementedError('2D or 3D only')

        super().__init__(**kwargs)
        

    def __call__(self, init, matO, matAff):
        """
        Parameters
            init: initialization translation, (dim 0 = batch dim).
            matO: orientation matrix, (no batch dim).
            matAff: estimated voxelic affine transformation, (dim 0 = batch dim).

        """
        
        matInit = tf.map_fn(self._single_trans2mat, init, dtype=tf.float32)
        if matAff.shape[1] == self.ndims:
            matAff = tf.map_fn(self._homogen_ext, matAff, dtype=tf.float32)
        matO = self._matO_perm(matO)
        
        matReal = tf.matmul(matO, matAff)
        matReal = tf.matmul(matReal, tf.linalg.inv(matO))
        matReal = tf.matmul(matInit, matReal)
        
        return matReal

    def _single_trans2mat(self, vector):
        
        vector = tf.concat((vector, [1]), axis=0)
        mat = tf.eye(self.ndims)
        mat = tf.concat((mat, tf.zeros((1, self.ndims))), axis=0)
        mat = tf.concat((mat, tf.expand_dims(vector, axis=1)), axis=1)  
            
        return mat

    def _homogen_ext(self, mat):
        
        extensionh = tf.zeros((1, self.ndims))
        extensionh = tf.concat((extensionh, [[1]]), axis=1)
        mat = tf.concat((mat, extensionh), axis=0)
        
        return mat
    
    def _matO_perm(self, matO):

        trans = tf.zeros((self.ndims,1))
        perm = tf.eye(self.ndims)[::-1]
        perm = tf.concat((perm, trans), axis=1)
        perm = self._homogen_ext(perm)
        
        if matO.shape[1] == self.ndims:
            matO = self._homogen_ext(matO)

        matO = tf.matmul(matO, perm)
        
        return matO
   

def one_hot_enc(seg, labs, segtype='itkimg', dtype=np.int8):
    """
    segtype can be itkimg or array
    """
    if segtype == 'itkimg':
        ndims = seg.GetDimension()
        origin = seg.GetOrigin() + (0,)
        direction = seg.GetDirection()
        direction = np.eye(ndims+1)
        direction[0:ndims,0:ndims] = np.reshape(seg.GetDirection(),[ndims,ndims])
        direction = np.ravel(direction)
        spacing = seg.GetSpacing() + (1,)
        seg = sitk.GetArrayFromImage(seg)
        
    seg = [seg==lab for lab in labs]
    seg = np.stack(seg, axis=-1)
    seg = seg.astype(dtype)
    
    if segtype == 'itkimg':    
        seg = np.transpose(seg, [ndims] + [*range(ndims)])
        seg = sitk.GetImageFromArray(seg, isVector=False)
        seg.SetOrigin(origin)
        seg.SetDirection(direction)
        seg.SetSpacing(spacing)
    
    return seg


def grid_img(volshape, omitdim=[2], spacing=5):
    g = np.zeros(volshape)
    
    for i in range(0,volshape[0], spacing):
        if 0 not in omitdim:
            g[i,:,:] = 1
    for j in range(0,volshape[1], spacing):
        if 1 not in omitdim:
            g[:,j,:] = 1
    for k in range(0,volshape[2], spacing):
        if 2 not in omitdim:
            g[:,:,k] = 1 
    return g


def quadratic_unidir_to_dense_shift(matrix, dire, shape, center=None, indexing='ij'):
    """
    Similar to voxelmorph.utils.affine_to_dense_shift(), but for a quadratic transfo.
    """

    if isinstance(shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        shape = shape.as_list()

    if not tf.is_tensor(matrix) or not matrix.dtype.is_floating:
        matrix = tf.cast(matrix, tf.float32)

    # check input shapes
    ndims = len(shape)
    if matrix.shape[-1] != ndims or matrix.shape[-2] != ndims:
        raise ValueError(f'Quadratic matrix must be squared batch_size x ndims x ndims (ndims={ndims}D).')

    # list of volume ndgrid
    # N-long list, each entry of shape
    mesh = ne.utils.volshape_to_meshgrid(shape, indexing=indexing)
    mesh = [f if f.dtype == matrix.dtype else tf.cast(f, matrix.dtype) for f in mesh]

    # transform into a large matrix
    flat_mesh = [tf.reshape(f, [-1]) for f in mesh]
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # ndims x nb_voxels
    if center is not None:
        center = tf.cast(center, dtype=matrix.dtype)[..., None]
        mesh_matrix -= center    
        
    # compute locations
    loc_matrix = tf.matmul(matrix, mesh_matrix)              # ndims x nb_voxels
    loc_matrix = loc_matrix * mesh_matrix                    # ndims x nb_voxels
    loc_matrix = tf.math.reduce_sum(loc_matrix, axis=0)      # nb_voxels
    loc = tf.reshape(loc_matrix, list(shape))                # *shape x 1

    # get shifts and return
    loc = [loc if d==dire else tf.zeros_like(loc) for d in range(ndims)]
    loc = tf.stack(loc, axis=-1)                             # *shape x ndims
    
    return loc

  
def integrate_svf(svf, int_steps=7, out_tr=True, alpha=1):
    
    ndims = svf.GetDimension()
    
    if alpha != 1:
        svf = sitk.Compose([sitk.VectorIndexSelectionCast(svf, d) * alpha for d in range(ndims)])
        
    # scaling
    svf = sitk.Compose([sitk.VectorIndexSelectionCast(svf, d)/(2**int_steps) for d in range(ndims)])

    # squaring
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetUseNearestNeighborExtrapolator(True)
    resampler.SetReferenceImage(svf)
    for _ in range(int_steps): 
        svf0 = copy.deepcopy(svf)
        transfo = sitk.DisplacementFieldTransform(svf)    
        resampler.SetTransform(transfo)
        svf = svf0 + resampler.Execute(svf0)
    
    if out_tr:
        return sitk.DisplacementFieldTransform(svf)
    else:
        return svf

