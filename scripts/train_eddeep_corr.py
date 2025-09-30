import os
import sys
eddeep_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(eddeep_dir)
import glob
import numpy as np
import tensorflow as tf    
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tqdm import trange
import pandas

import eddeep

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))       
        
parser = argparse.ArgumentParser(description="Training script for the image translation part of Eddeep.")

# training and validation data, pre-trained translator
parser.add_argument('-t', '--train_data', type=str, required=True, help='Path to the raw training data following nested subfolder structure below.')
parser.add_argument('-v', '--val_data', type=str, required=False, default=None, help='Path to the raw validation data following nested subfolder structure below.')
# ├── sub_001
# │   ├── AP
# │   │   ├── b0
# │   │   │   ├── vol_dir1.nii.gz
# │   │   │   ├── vol_dir2.nii.gz
# │   │   │   ├── ...
# │   │   ├── b1000
# │   │   │   ├── ...
# │   │   ├── ...
# │   │   ├── vol_b2000_mean.nii.gz (only for translation)
# │   │   ├── ...
# │   └── PA
# │       ├── ...
# ├── sub_002
# │   ├── ..._mini/corr_test_mini -r 1
# ├── ...
parser.add_argument('-tr', '--trans', type=str, required=True, help="Path to the pre-trained image translation model. Required.")
parser.add_argument('-k', '--kpad', type=int, required=False, default=5, help='k to pad the input so that its shape is of form 2**k. Has to be >= number encoding steps for deformable transfo, and equal to the one used for the translator.')
# distortion constraints
parser.add_argument('-p', '--ped', type=int, required=True, help='Axis number of the phase encoding directions (starting at 0). Usually 1 for AP/PA. Required.')
parser.add_argument('-trsf', '--transfo', type=str, required=False, default='quadratic', help="Type of geometric transformation for the distortion ('linear', 'quadratic' or 'deformable'). Default: 'quadratic'.")
# model and its hyper-paramaters
parser.add_argument('-o', '--model', type=str, required=True, help="Path prefix to the output model (without extension). Required.")
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-4, help="Learning rate. Default: 1e-4.")
parser.add_argument('-enc', '--enc-nf', type=int, nargs='+', required=False, default=[16,28,56,75,128], help="Number of encoder features. Default: 16 28 56 75 128.")
parser.add_argument('-dec', '--dec-nf', type=int, nargs='+', required=False, default=[128,75,56,26,16,16], help="Number of decoder features (only for deformable transfo). Default: 128 75 56 26 16 16.")
parser.add_argument('-den', '--dense-nf', type=int, nargs='+', required=False, default=[128,64], help="Number of MLP features. Default: 64.")
parser.add_argument('-nconv', '--nb-conv-lvl', type=int, required=False, default=1, help="Number of convolutions per level for the generator. Default: 1.")
parser.add_argument('-e', '--epochs', type=int, required=True, help="Number of epochs. Required.")
parser.add_argument('-b', '--batch-size', type=int, required=False, default=1, help='Batch size. Default: 1.')
parser.add_argument('-l', '--loss', type=str, required=False, default='l2', help="Loss between translated corrected DW image and translated b=0 image ('l1' or 'l2'). Default: 'l2'.")
parser.add_argument('-wr', '--weight-regul', type=float, required=False, default=0.05, help="Weight of the regularisation loss (only for deformable transfo). Default: 0.05.")
# augmentation
parser.add_argument('-as', '--aug_spat_prob', type=float, required=False, default=0, help='Probability of performing spatial augmentation. Default: 0.')
# other
parser.add_argument('-r', '--resume', type=int, required=False, default=0, help='Resume a training that stopped for some reason (1: yes, 0: no). Default: 0.')
parser.add_argument('-seed', '--seed', type=int, required=False, default=None, help='Seed for randomness. Default: None.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

args.resume = bool(args.resume)
is_deformable = args.transfo == 'deformable'
if not is_deformable: args.dec_nf = []

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

os.makedirs(os.path.join(args.model + '_imgs'), exist_ok=True)

with open(args.model + '_args.txt', 'w') as file:
    for arg in vars(args):
        file.write("{}: {}\n".format(arg, getattr(args, arg)))
      
#%% Generators to access training (and validation) data.

is_val = args.val_data is not None

# training images
sub_dirs = sorted(glob.glob(os.path.join(args.train_data, '*', '')))
random.shuffle(sub_dirs)
n_train = len(sub_dirs)
gen_train = eddeep.generators.eddeep_fromDWI(subdirs=sub_dirs,
                                             k=args.kpad,
                                             spat_aug_prob=args.aug_spat_prob,
                                             aug_dire=args.ped,
                                             batch_size=args.batch_size)

# validation images
if not is_val:  
    gen_val = None
    sample = next(gen_train)
else:
    sub_dirs_val = sorted(glob.glob(os.path.join(args.val_data, '*', '')))
    random.shuffle(sub_dirs_val)
    n_val = len(sub_dirs_val)
    gen_val = eddeep.generators.eddeep_fromDWI(subdirs=sub_dirs_val,                                                          
                                               k=args.kpad,
                                               spat_aug_prob=0,
                                               batch_size=args.batch_size)
    sample = next(gen_val)
    

eddeep.utils.develop(sample)
imshape = sample[0].shape[1:-1]
sl_sag = int(sample[0].shape[3]*0.45)
sl_axi = int(sample[0].shape[1]*0.45)


#%% Prepare and build the model

loss_file = args.model + '_losses.csv'

model_path = args.model + '_best.keras'
model_last_path = args.model + '_last.keras'
    
translator = tf.keras.models.load_model(args.trans)

if args.resume:
    # load existing model
    registrator = tf.keras.models.load_model(model_last_path)
    tab_loss = pandas.read_csv(loss_file, sep=',')
    if is_val:
        if is_deformable: best_loss = tab_loss.val_img_loss + args.weight_regul * tab_loss.val_reg_loss
        else: best_loss = np.min(tab_loss.val_img_loss)
    else:
        if is_deformable: best_loss = tab_loss.img_loss + args.weight_regul * tab_loss.reg_loss
        else: best_loss = np.min(tab_loss.img_loss)
    initial_epoch = tab_loss.epoch.iloc[-1]
    print('resuming training at epoch: ' + str(initial_epoch))

else:
    # build the model
    registrator = eddeep.models.eddy_reg(imshape=imshape,
                                         ped=args.ped,
                                         nb_enc_feats=args.enc_nf,
                                         nb_dec_feats=args.dec_nf,
                                         transfo=args.transfo,               
                                         jacob_mod=False,
                                         nb_dense_feats=args.dense_nf,
                                         nb_conv_lvl=args.nb_conv_lvl)
    tf.keras.utils.plot_model(registrator, to_file=args.model + '_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)

    if args.loss == 'l1': img_loss_fun = tf.keras.losses.MeanAbsoluteError()
    elif args.loss == 'l2': img_loss_fun = tf.keras.losses.MeanSquaredError()
    loss_fun = [img_loss_fun]
    if is_deformable:
        reg_loss_fun = eddeep.losses.regul_unidir(p=2, order=1).loss
        loss_fun += [reg_loss_fun]
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    registrator.compile(optimizer=optimizer, loss=loss_fun)

    try: os.remove(loss_file)
    except OSError: pass
    f = open(loss_file,'w')
    f.write('epoch,img_loss') 
    if is_deformable: f.write(',reg_loss')
    if is_val: 
        f.write(',val_img_loss')
        if is_deformable: f.write(',val_reg_loss')
    f.write('\n')
    f.close()
    
    initial_epoch = 0
    best_loss = np.Inf
        
    
#%% Train the model

n_train_steps = np.max((1, n_train // args.batch_size))
if is_val:
    n_val_steps = np.max((1, n_val // args.batch_size))
n_substeps = 20    # a bit arbitrary, so that each b-value has a chance to pop a few times

y_b0 = translator(sample[0], training=False)
y_dw = translator(sample[1], training=False)

for epoch in range(initial_epoch, args.epochs):   
    
    print('epoch: %d/%d' % (epoch+1, args.epochs))
    if is_deformable:
        v_loss_epoch = np.zeros(2)
        v_loss_val_epoch = np.zeros(2)
    else:
        v_loss_epoch = np.zeros(1)
        v_loss_val_epoch = np.zeros(1)       
    
    for _ in trange(n_train_steps, desc='train'):
        for substep in range(n_substeps):
            
            x0, x = next(gen_train)
            
            v_loss = eddeep.training.train_corr_step(x0=x0, x=x, 
                                                     translator=translator, registrator=registrator,
                                                     deformable=is_deformable, reg_loss_weight=args.weight_regul)
            v_loss_epoch += v_loss.numpy()

    v_loss_epoch /= n_train_steps * n_substeps
    
    if is_val:
        for _ in trange(n_val_steps, desc='val  '):
            for substep in range(n_substeps):

                x0, x = next(gen_val)          

                v_loss_val = eddeep.training.val_corr_step(x0=x0, x=x, 
                                                           translator=translator, registrator=registrator,
                                                           deformable=is_deformable, reg_loss_weight=args.weight_regul)
                v_loss_val_epoch += v_loss_val.numpy()
            
        v_loss_val_epoch /= n_val_steps * n_substeps
        

    f = open(loss_file,'a')
    if is_deformable: print('  train | img: %.3e, reg: %.3e' % tuple(v_loss_epoch))
    else: print('  train | img: %.3e' % v_loss_epoch)
    if is_val: 
        if is_deformable: print('  val   | img: %.3e, reg: %.3e ' % tuple(v_loss_val_epoch))
        else: print('  val   | img: %.3e' % v_loss_val_epoch)
    
    f = open(loss_file,'a')
    f.write(str(epoch+1) + ',' + ','.join(map(str, v_loss_epoch)))
    if is_val: f.write(',' + ','.join(map(str, v_loss_val_epoch)))
    f.write('\n')
    f.close()
    
    if is_deformable:
        if is_val: v_corr_loss_epoch = v_loss_val_epoch[0] + args.weight_regul * v_loss_val_epoch[1]
        else: v_corr_loss_epoch = v_loss_epoch[0] + args.weight_regul * v_loss_epoch[1]
    else:
        if is_val: v_corr_loss_epoch = v_loss_val_epoch[0]
        else: v_corr_loss_epoch = v_loss_epoch[0]

    if v_corr_loss_epoch < best_loss:
        best_loss = v_corr_loss_epoch
        registrator.save(model_path)
    registrator.save(model_last_path )
    
    y_dw_corr = registrator([y_b0, y_dw], training=False)
    if args.transfo == 'deformable':
        y_dw_corr = y_dw_corr[0]

    f, axs = plt.subplots(2,5); f.dpi = 300
    plt.subplots_adjust(wspace=0.01,hspace=-0.1)

    axs[0][0].imshow(np.fliplr((y_dw[0,:,:,sl_sag,0]-y_b0[0,:,:,sl_sag,0]))**2, vmin=0, vmax=0.005, origin="lower")
    axs[0][1].imshow(np.fliplr(y_dw[0,:,:,sl_sag,0]), vmin=0, vmax=1, origin="lower")
    axs[0][2].imshow(np.fliplr(y_b0[0,:,:,sl_sag,0]), vmin=0, vmax=1, origin="lower")
    axs[0][3].imshow(np.fliplr(y_dw_corr[0,:,:,sl_sag,0]), vmin=0, vmax=1, origin="lower")
    axs[0][4].imshow(np.fliplr((y_dw_corr[0,:,:,sl_sag,0]-y_b0[0,:,:,sl_sag,0]))**2, vmin=0, vmax=0.005, origin="lower")
    
    axs[1][0].imshow(np.fliplr((y_dw[0,sl_axi,:,:,0]-y_b0[0,sl_axi,:,:,0]))**2, vmin=0, vmax=0.005, origin="lower")
    axs[1][1].imshow(np.fliplr(y_dw[0,sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
    axs[1][2].imshow(np.fliplr(y_b0[0,sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
    axs[1][3].imshow(np.fliplr(y_dw_corr[0,sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
    axs[1][4].imshow(np.fliplr((y_dw_corr[0,sl_axi,:,:,0]-y_b0[0,sl_axi,:,:,0]))**2, vmin=0, vmax=0.005, origin="lower")

    axs[0][0].set_title('sq diff', fontsize=7)
    axs[0][1].set_title('trans dw', fontsize=7)
    axs[0][2].set_title('trans b=0 (ref)', fontsize=7)
    axs[0][3].set_title('trans dw corr', fontsize=7)
    axs[0][4].set_title('sq diff corr', fontsize=7)
    
    for i in range(2): 
        for j in range(5): axs[i][j].axis('off')
    plt.suptitle('epoch: ' + str(epoch+1), ha='center', y=0.89, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model + '_imgs','img_' + str(epoch) + '.png'), bbox_inches='tight', dpi=300)        
    plt.close()

eddeep.utils.plot_losses(loss_file, is_val=is_val)

