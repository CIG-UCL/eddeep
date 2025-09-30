import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI

from eddeep import layers
from external import voxelmorph


def get_conv(ndims):
    if ndims == 3: return KL.Conv3D
    elif ndims == 2: return KL.Conv2D

def get_upsample(ndims):
    if ndims == 3: return KL.UpSampling3D
    elif ndims == 2: return KL.UpSampling2D

def get_maxpool(ndims):
    if ndims == 3: return KL.MaxPool3D
    elif ndims == 2: return KL.MaxPool2D
    

def cnn(imshape,
        nb_in_chan=1,
        nb_out_chan=1,
        nb_enc_feats=None,
        nb_dec_feats=None,
        nb_bottleneck_feats=None,
        nb_conv_lvl=1,
        do_skips=True,
        down_type='max',    # 'conv' or 'max'
        final_activation=None,
        activation='leaky_relu',
        res_factor=2,
        ker_size=3,
        get_bottle=False,
        name='cnn'):
    
    if nb_enc_feats is None: nb_enc_feats = [16, 32, 64, 128]
    if nb_dec_feats is None: nb_dec_feats = [128, 64, 32, 16]
    if nb_bottleneck_feats is None: nb_bottleneck_feats = []
       
    # Define model input
    x_in = tf.keras.Input(shape=(*imshape, nb_in_chan), name=f'{name}_input')
    x = x_in

    ndims = len(imshape)
    conv = get_conv(ndims)
    upsample = get_upsample(ndims)
    maxpool = get_maxpool(ndims)

    if len(nb_dec_feats) == 0:
        do_skips = False
    skips = []

    # encoder
    for i, f in enumerate(nb_enc_feats):
        for c in range(nb_conv_lvl):
            x = conv(filters=f, kernel_size=ker_size, strides=1, padding='same',
                     activation=activation, name=f'{name}_enc_l{i}_conv{c}')(x)
        if do_skips:
            skips.append(x)
        if down_type == 'conv':
            x = conv(filters=f, kernel_size=ker_size, strides=res_factor, padding='same',
                     activation=activation, name=f'{name}_enc_l{i}_down')(x)
        elif down_type == 'max':
            x = maxpool(res_factor, name=f'{name}_enc_l{i}_down')(x)

    # bottleneck
    for i, f in enumerate(nb_bottleneck_feats):
        for c in range(nb_conv_lvl):
            x = conv(filters=f, kernel_size=ker_size, strides=1, padding='same',
                     activation=activation, name=f'{name}_bottleneck_l{i}_conv{c}')(x)
    if get_bottle:
        x_bottle = x

    # decoder
    skips.reverse()
    for i, f in enumerate(nb_dec_feats):
        for c in range(nb_conv_lvl):
            x = conv(filters=f, kernel_size=ker_size, strides=1, padding='same',
                     activation=activation, name=f'{name}_dec_l{i}_conv{c}')(x)
        if i < len(nb_enc_feats):
            x = upsample(res_factor, name=f'{name}_enc_l{i}_up')(x)
            if do_skips:
                x = KL.Concatenate(axis=-1, name=f'{name}_dec_concat_l{i}')([x, skips[i]])

    # final
    x = conv(filters=nb_out_chan, kernel_size=ker_size, strides=1, padding='same',
             activation=final_activation, name=f'{name}_last_conv')(x)

    if get_bottle:
        model = tf.keras.Model(inputs=x_in, outputs=[x, x_bottle], name=name)
    else:
        model = tf.keras.Model(inputs=x_in, outputs=x, name=name)

    return model



def eddy_reg(imshape,
             ped,
             nb_enc_feats=None,
             nb_dec_feats=None,        # only used if transfo is 'deformable'
             transfo='quadratic',      # 'linear', 'quadratic' or 'deformable'
             nb_conv_lvl=1,
             down_type='max',
             jacob_mod=False,
             nb_dense_feats=None,
             activation='leaky_relu',
             name='eddy_reg'):
    
    if nb_enc_feats is None: nb_enc_feats = [16,28,56,75,128]
    if nb_dec_feats is None: nb_dec_feats = [128,75,56,26,16,16]
    if nb_dense_feats is None: nb_dense_feats = [128,64]
    
    ndims = len(imshape)
    center = [shape // 2 for shape in imshape]
    conv = get_conv(ndims)
    
    trans_init = KI.RandomNormal(stddev=1e-2)
    lin_init = KI.RandomNormal(stddev=1e-3)  
    quad_init = KI.RandomNormal(stddev=1e-5) 
        
    b0_in = KL.Input(shape=(*imshape, 1), name='input_b0')
    dw_in = KL.Input(shape=(*imshape, 1), name='input_dw') 
    b0_dw = KL.Concatenate(axis=-1, name='inputs_concat')([b0_in, dw_in])
    
    if transfo in ('linear', 'quadratic'):
        
        cnn_model = cnn(imshape=imshape, 
                        nb_in_chan=2, 
                        nb_out_chan=nb_enc_feats[-1],
                        nb_enc_feats=nb_enc_feats, 
                        nb_dec_feats=[],
                        nb_conv_lvl=nb_conv_lvl, 
                        down_type=down_type,
                        activation=activation,
                        final_activation=activation, 
                        name='eddy_cnn')
        
        transfo_params = cnn_model(b0_dw)

        transfo_params = KL.Flatten()(transfo_params)   
        for j, nf in enumerate(nb_dense_feats):
            transfo_params = KL.Dense(nf, activation=activation, name='mlp_eddy_%d' % j)(transfo_params)
        
        trans_eddy = transfo_params
        lin_eddy = transfo_params            
        trans_rig = transfo_params
        lin_rig = transfo_params
    
        trans_eddy = KL.Dense(1, kernel_initializer=trans_init, name='trans_eddy',activation=None)(trans_eddy)
        lin_eddy  = KL.Dense(ndims, kernel_initializer=lin_init, name='lin_eddy',activation=None)(lin_eddy)
        transfo_eddy = layers.AffCoeffToMatrix(ndims=ndims, dire=ped, transfo_type='dir_affine', center=center, name='build_eddy_affine_transfo')([trans_eddy,lin_eddy])
        transfo_eddy = voxelmorph.layers.AffineToDenseShift(shape=imshape, shift_center=False)(transfo_eddy)
        if transfo == 'quadratic':
            quad_eddy = transfo_params
            quad_eddy = KL.Dense(ndims*(ndims+1)//2, kernel_initializer=quad_init, name='quad_eddy',activation=None)(quad_eddy)
            transfo_quad_eddy = layers.QuadCoeffToMatrix(ndims=ndims, name='build_eddy_quad_transfo')(quad_eddy)
            transfo_quad_eddy = layers.QuadUnidirToDenseShift(shape=imshape, dire=ped, center=center)(transfo_quad_eddy)
            transfo_eddy = KL.add((transfo_eddy, transfo_quad_eddy))
            
    elif transfo == 'deformable':
        # eddy part
        cnn_model = cnn(imshape=imshape,
                        nb_in_chan=2,
                        nb_out_chan=1,
                        nb_enc_feats=nb_enc_feats,
                        nb_dec_feats=nb_dec_feats,
                        nb_conv_lvl=nb_conv_lvl,
                        down_type=down_type,
                        activation = activation,
                        final_activation=None,
                        get_bottle=True,
                        name='eddy_cnn')

        transfo_unidir_eddy, bottle = cnn_model(b0_dw)
        
        transfo_eddy = layers.expand_unidir_shift(ndims=ndims, ped=ped)(transfo_unidir_eddy)

        # rigid part
        rigid_params = conv(filters=nb_enc_feats[-1], kernel_size=3, strides=1, padding='same',
                            activation=activation, name='eddy_cnn_enc_last_conv')(bottle)
        rigid_params = KL.Flatten()(rigid_params)
        for j, nf in enumerate(nb_dense_feats):
            rigid_params = KL.Dense(nf, activation=activation, name='mlp_eddy_%d' % j)(rigid_params) 
        
        trans_rig = rigid_params
        lin_rig = rigid_params
            
    trans_rig = KL.Dense(ndims, kernel_initializer=trans_init, name='trans_rig',activation=None)(trans_rig)
    lin_rig  = KL.Dense(ndims, kernel_initializer=lin_init, name='lin_rig',activation=None)(lin_rig)    
    transfo_rig = layers.AffCoeffToMatrix(ndims=ndims, transfo_type='rigid', center=center, name='build_rigid_transfo')([trans_rig,lin_rig])

    full_transfo = voxelmorph.layers.ComposeTransform(shift_center=False, name='compose_transfos')([transfo_eddy, transfo_rig])

    dw_corr = voxelmorph.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='full_transformer')([dw_in, full_transfo])
    if jacob_mod:
        dw_corr = layers.JacobianMultiplyIntensities(indexing='ij', is_shift=True, name='jac_modul_dw')([dw_corr, full_transfo])
    
    outputs = [dw_corr]
    if transfo == 'deformable':
        outputs += [transfo_unidir_eddy]
    
    model = tf.keras.Model(inputs=[b0_in, dw_in], outputs=outputs, name=name)
    return model
