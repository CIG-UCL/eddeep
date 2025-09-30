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

# training and validation data
parser.add_argument('-t', '--train_data', type=str, required=True, help='Path to the eddy-corrected training data following nested subfolder structure below.')
parser.add_argument('-v', '--val_data', type=str, required=False, default=None, help='Path to the eddy-corrected validation data following nested subfolder structure below.')
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
# │   ├── ...
# ├── ...
parser.add_argument('-B', '--target_bval', type=int, required=True, help='Target b-value for translation (must be among the existing b-values in the data). Required.')
parser.add_argument('-k', '--kpad', type=int, required=False, default=5, help='k to pad the input so that its shape is of form 2**k. Has to be >= number encoding steps. Default: 5')
# model and its hyper-paramaters
parser.add_argument('-o', '--model', type=str, required=True, help="Path prefix to the output model (without extension). Required.")
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-4, help="Learning rate. Default: 1e-4.")
parser.add_argument('-denc', '--dis-enc-nf', type=int, nargs='+', required=False, default=[16,32,64], help="Number of encoder features for the discriminator. Default: 16 32 64.")
parser.add_argument('-genc', '--gen-enc-nf', type=int, nargs='+', required=False, default=[16,32,64,128], help="Number of encoder features for the generator. Default: 16 32 64 128.")
parser.add_argument('-gdec', '--gen-dec-nf', type=int, nargs='+', required=False, default=[128,64,32,16], help="Number of decoder features for the generator. Default: 128 64 32 16.")
parser.add_argument('-nconv', '--nb-conv-lvl', type=int, required=False, default=1, help="Number of convolutions per level for the generator. Default: 1.")
parser.add_argument('-e', '--epochs', type=int, required=True, help="Number of epochs. Required.")
parser.add_argument('-b', '--batch-size', type=int, required=False, default=1, help='Batch size. Default: 1.')
parser.add_argument('-gan', '--gan', type=int, required=False, default=1, help='Use GAN or simple U-Net (1: GAN, 0: simple U-Net). Default: 1.')
parser.add_argument('-l', '--loss', type=str, required=False, default='l1', help="Loss between generated and real image ('l1' or 'l2'). Default: 'l1'.")
parser.add_argument('-wi', '--img-loss-weight', type=float, required=False, default=100., help="Weight for the image reconstruction loss (only for GAN). Default: 100.")
# augmentation
parser.add_argument('-as', '--aug_spat_prob', type=float, required=False, default=0, help='Probability of performing spatial augmentation. Default: 0.')
parser.add_argument('-ai', '--aug_int_prob', type=float, required=False, default=0, help='Probability of performing intensity augmentation. Default: 0.')
# other
parser.add_argument('-r', '--resume', type=int, required=False, default=0, help='Resume a training that stopped for some reason (1: yes, 0: no). Default: 0.')
parser.add_argument('-seed', '--seed', type=int, required=False, default=None, help='Seed for randomess. Default: None.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args.resume = bool(args.resume)
args.gan = bool(args.gan)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

os.makedirs(os.path.join(args.model) + '_imgs', exist_ok=True)

with open(args.model + '_args.txt', 'w') as file:
    for arg in vars(args):
        file.write("{}: {}\n".format(arg, getattr(args, arg)))
      
#%% Generators to access training (and validation) data.

is_val = args.val_data is not None

# training images
sub_dirs = sorted(glob.glob(os.path.join(args.train_data, '*', '')))
random.shuffle(sub_dirs)

gen_train = eddeep.generators.eddeep_fromDWI(subdirs=sub_dirs,
                                             k=args.kpad,
                                             target_bval = args.target_bval,
                                             get_dwmean = True,
                                             spat_aug_prob = args.aug_spat_prob,
                                             int_aug_prob = args.aug_int_prob,
                                             batch_size=args.batch_size)

n_train = len(sub_dirs)

# validation images
if not is_val:  
    gen_val = None
    sample = next(gen_train)
else:
    sub_dirs_val = sorted(glob.glob(os.path.join(args.val_data, '*', '')))
    gen_val = eddeep.generators.eddeep_fromDWI(subdirs=sub_dirs_val,                                                          
                                               k=args.kpad,
                                               target_bval = args.target_bval,
                                               get_dwmean = True,
                                               spat_aug_prob = 0,
                                               int_aug_prob = 0,
                                               batch_size=args.batch_size)
    n_val = len(sub_dirs_val)
    sample = next(gen_val)
    

eddeep.utils.develop(sample)
inshape = sample[0].shape[1:-1]
sl_sag=int(sample[0].shape[3]*0.45)
sl_axi=int(sample[0].shape[1]*0.45)


#%% Prepare and build the model

loss_file = args.model + '_losses.csv'
gen_path = args.model + '_gen_best.keras'
gen_last_path = args.model + '_gen_last.keras'
if args.gan:
    dis_path = args.model + '_dis_best.keras'
    dis_last_path = args.model + '_dis_last.keras'
else:
    discriminator = None
down_type = 'max'

if args.resume:
    # load existing model
    tab_loss = pandas.read_csv(loss_file, sep=',')
    generator = tf.keras.models.load_model(gen_last_path)
    if args.gan:
        discriminator = tf.keras.models.load_model(dis_last_path)
        if is_val: best_gen_loss = np.min(tab_loss.val_gen0 + tab_loss.val_gen) / 2
        else: best_gen_loss = np.min(tab_loss.gen0 + tab_loss.gen) / 2
    else:
        if is_val: best_gen_loss = np.min(tab_loss.val_img0 + tab_loss.val_img) / 2
        else: best_gen_loss = np.min(tab_loss.img0 + tab_loss.img) / 2
        
    initial_epoch = tab_loss.epoch.iloc[-1]
    print('resuming training at epoch: ' + str(initial_epoch))

else:
    # build the models
    generator = eddeep.models.cnn(imshape=inshape,
                                  nb_in_chan=1, nb_out_chan=1,
                                  nb_enc_feats=args.gen_enc_nf, nb_dec_feats=args.gen_dec_nf, nb_bottleneck_feats=[],
                                  nb_conv_lvl=args.nb_conv_lvl, do_skips=True,
                                  down_type=down_type, final_activation=None)
    tf.keras.utils.plot_model(generator, to_file=args.model + '_gen_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)
    
    if args.loss == 'l1': img_loss_fun = tf.keras.losses.MeanAbsoluteError()
    elif args.loss == 'l2': img_loss_fun = tf.keras.losses.MeanSquaredError()
    gen_opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    generator.compile(optimizer=gen_opt, loss=img_loss_fun)
    
    if args.gan:
        discriminator = eddeep.models.cnn(imshape=inshape,
                                          nb_in_chan=2, nb_out_chan=1,
                                          nb_enc_feats=args.dis_enc_nf, nb_dec_feats=[], nb_bottleneck_feats=[],
                                          nb_conv_lvl=1, down_type=down_type, 
                                          final_activation='sigmoid')
        tf.keras.utils.plot_model(discriminator, to_file=args.model + '_dis_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)
        
        dis_loss_fun = tf.keras.losses.MeanSquaredError()
        dis_opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        discriminator.compile(optimizer=dis_opt, loss=dis_loss_fun)
        
    try: os.remove(loss_file)
    except OSError: pass
    f = open(loss_file,'w')
    if args.gan:
        f.write('epoch,dis0_real,dis_real,dis0_fake,dis_fake,gen0,gen,img0,img')
        if is_val: f.write(',val_dis0_real,val_dis_real,val_dis0_fake,val_dis_fake,val_gen0,val_gen,val_img0,val_img') 
    else:
        f.write('epoch,img0,img') 
        if is_val: f.write(',val_img0,val_img')
    f.write('\n')
    f.close()

    initial_epoch = 0
    best_gen_loss = np.Inf
    

#%% Train the model

n_train_steps = n_train // args.batch_size
if is_val:
    n_val_steps = n_val // args.batch_size
n_substeps = 10    # a bit arbitrary, so that each b-value has a chance to pop a few times

adversarial = tf.constant(args.gan)
img_loss_weight = tf.constant(args.img_loss_weight)

if args.gan: 
    train_trans_step = eddeep.training.train_transGAN_step
    val_trans_step = eddeep.training.val_transGAN_step
else:
    train_trans_step = eddeep.training.train_trans_step
    val_trans_step = eddeep.training.val_trans_step
    
    
for epoch in range(initial_epoch, args.epochs):
    
    print('epoch: %d/%d' % (epoch+1, args.epochs))
    if args.gan:
        v_loss_epoch = np.zeros(8)
        v_loss_val_epoch = np.zeros(8)
    else:
        v_loss_epoch = np.zeros(2)
        v_loss_val_epoch = np.zeros(2)        
    
    for _ in trange(n_train_steps, desc='train'):
        for _ in range(n_substeps):

            x0, x, y = next(gen_train)
            
            v_loss = train_trans_step(x0=x0, x=x, y=y, 
                                      gen=generator, dis=discriminator, 
                                      img_loss_weight=img_loss_weight)
            v_loss_epoch += v_loss

    v_loss_epoch /= n_train_steps * n_substeps
        
    if is_val:
        for _ in trange(n_val_steps, desc='val  '):
            for _ in range(n_substeps):

                x0, x, y = next(gen_val)
                
                v_loss_val = val_trans_step(x0=x0, x=x, y=y, 
                                            gen=generator, dis=discriminator,
                                            img_loss_weight=img_loss_weight)
                v_loss_val_epoch += v_loss_val
    
        v_loss_val_epoch /= n_val_steps * n_substeps

    if args.gan:
        print('  train (b=0, dw) | dis_real (%.3f, %.3f), dis_fake: (%.3f, %.3f), gen: (%.3f, %.3f), img: (%.3e, %.3e)' % tuple(v_loss_epoch))
    else:
        print('  train (b=0, dw) | img: (%.3e, %.3e)' % tuple(v_loss_epoch))
    if is_val:
        if args.gan:
            print('  val   (b=0, dw) | dis_real (%.3f, %.3f), dis_fake: (%.3f, %.3f), gen: (%.3f, %.3f), img: (%.3e, %.3e)' % tuple(v_loss_val_epoch))
        else:
            print('  val   (b=0, dw) | img: (%.3e, %.3e)' % tuple(v_loss_val_epoch))
    
    f = open(loss_file,'a')
    f.write(str(epoch+1) + ',' + ','.join(map(str, v_loss_epoch)))
    if is_val:
        f.write(',' + ','.join(map(str, v_loss_val_epoch)))
    f.write('\n')
    f.close()

    if args.gan:
        if is_val: v_gen_loss_epoch = (v_loss_val_epoch[4] + v_loss_val_epoch[5]) / 2
        else: v_gen_loss_epoch = (v_loss_epoch[4] + v_loss_epoch[5]) / 2
    else:
        if is_val: v_gen_loss_epoch = (v_loss_val_epoch[0] + v_loss_val_epoch[1]) / 2
        else: v_gen_loss_epoch = (v_loss_epoch[0] + v_loss_epoch[1]) / 2
    
    if v_gen_loss_epoch < best_gen_loss:
        best_gen_loss = v_gen_loss_epoch
        generator.save(gen_path)
        if args.gan:
            discriminator.save(dis_path)
    generator.save(gen_last_path)
    if args.gan:
        discriminator.save(dis_last_path)

    y0_fake = generator(sample[0])
    y_fake = generator(sample[1])
    
    f, axs = plt.subplots(2,5); f.dpi = 500
    plt.subplots_adjust(wspace=0.01,hspace=-0.58)
    
    axs[0][0].imshow(np.fliplr(sample[0][0,:,:,sl_sag,0]), vmin=0, vmax=1, origin="lower")
    axs[0][0].axis('off'); axs[0][0].set_title('b=0', fontsize=7)
    axs[0][1].imshow(np.fliplr(y0_fake[0,:,:,sl_sag,0]), vmin=0, vmax=1, origin="lower")
    axs[0][1].axis('off'); axs[0][1].set_title('fake from b=0', fontsize=7)
    axs[0][2].imshow(np.fliplr(sample[2][0,:,:,sl_sag,0]), vmin=0, vmax=1, origin="lower")
    axs[0][2].axis('off'); axs[0][2].set_title('target', fontsize=7)
    axs[0][3].imshow(np.fliplr(y_fake[0,:,:,sl_sag,0]), vmin=0, vmax=1, origin="lower")
    axs[0][3].axis('off'); axs[0][3].set_title('fake from dw', fontsize=7)
    axs[0][4].imshow(np.fliplr(sample[1][0,:,:,sl_sag,0]), vmin=0, vmax=1, origin="lower")
    axs[0][4].axis('off'); axs[0][4].set_title('dw', fontsize=7)

    axs[1][0].imshow(np.fliplr(sample[0][0,sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
    axs[1][0].axis('off'); 
    axs[1][1].imshow(np.fliplr(y0_fake[0,sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
    axs[1][1].axis('off'); 
    axs[1][2].imshow(np.fliplr(sample[2][0,sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
    axs[1][2].axis('off'); 
    axs[1][3].imshow(np.fliplr(y_fake[0,sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
    axs[1][3].axis('off'); 
    axs[1][4].imshow(np.fliplr(sample[1][0,sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
    axs[1][4].axis('off'); 
    
    plt.suptitle('epoch: ' + str(epoch+1), ha='center', y=0.8, fontsize=8)

    plt.savefig(os.path.join(args.model + '_imgs','img_' + str(epoch) + '.png'), bbox_inches='tight',dpi=300)
    plt.close()

eddeep.utils.plot_losses(loss_file, is_val=is_val, nb_rows=2)

