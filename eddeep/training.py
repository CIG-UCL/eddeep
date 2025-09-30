import tensorflow as tf


@tf.function
def train_transGAN_step(x0, x, y, 
                     gen, dis=None, img_loss_weight=None):
    
    gen_opt = gen.optimizer
    img_loss_fun = gen.loss

    dis_opt = dis.optimizer
    dis_loss_fun = dis.loss

    batch_size = tf.shape(x0)[0]
    patch_shape = dis.output_shape[1:-1]
    real = tf.ones((batch_size, *patch_shape, 1))
    fake = tf.zeros((batch_size, *patch_shape, 1))

    with tf.GradientTape() as dis_tape, tf.GradientTape() as gen_tape:

        yhat0 = gen(x0, training=True)
        yhat = gen(x, training=True)

        dis0_real = dis(tf.concat([x0, y], axis=-1), training=True)
        dis0_fake = dis(tf.concat([x0, yhat0], axis=-1), training=True)
        dis0_real_loss = dis_loss_fun(real, dis0_real)
        dis0_fake_loss = dis_loss_fun(fake, dis0_fake)
        dis0_loss = (dis0_real_loss + dis0_fake_loss) / 2

        dis_real = dis(tf.concat([x, y], axis=-1), training=True)
        dis_fake = dis(tf.concat([x, yhat], axis=-1), training=True)
        dis_real_loss = dis_loss_fun(real, dis_real)
        dis_fake_loss = dis_loss_fun(fake, dis_fake)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2

        dis_loss_tot = (dis0_loss + dis_loss) / 2

        gen0_adv_loss = dis_loss_fun(real, dis0_fake)
        gen0_img_loss = img_loss_fun(y, yhat0)
        gen0_loss = gen0_adv_loss + img_loss_weight * gen0_img_loss

        gen_adv_loss = dis_loss_fun(real, dis_fake)
        gen_img_loss = img_loss_fun(y, yhat)
        gen_loss = gen_adv_loss + img_loss_weight * gen_img_loss

        gen_loss_tot = (gen0_loss + gen_loss) / 2


    dis_grads = dis_tape.gradient(dis_loss_tot, dis.trainable_variables)
    dis_opt.apply_gradients(zip(dis_grads, dis.trainable_variables))
    
    gen_grads = gen_tape.gradient(gen_loss_tot, gen.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, gen.trainable_variables))

    return (dis0_real_loss, dis_real_loss, dis0_fake_loss, dis_fake_loss,
            gen0_loss, gen_loss, gen0_img_loss, gen_img_loss)
    
    

@tf.function
def train_trans_step(x0, x, y, 
                     gen, dis=None, img_loss_weight=None):
    
    gen_opt = gen.optimizer
    img_loss_fun = gen.loss

    with tf.GradientTape() as gen_tape:

        yhat0 = gen(x0, training=True)
        img0_loss = img_loss_fun(y, yhat0)

        yhat = gen(x, training=True)
        img_loss = img_loss_fun(y, yhat)

        img_loss_tot = (img0_loss + img_loss) / 2
        
    gen_grads = gen_tape.gradient(img_loss_tot, gen.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, gen.trainable_variables))

    return (img0_loss, img_loss)



@tf.function
def val_transGAN_step(x0, x, y, 
                   gen, dis=None, img_loss_weight=None):
    
    img_loss_fun = gen.loss
    dis_loss_fun = dis.loss

    batch_size = tf.shape(x0)[0]
    patch_shape = dis.output_shape[1:-1]
    real = tf.ones((batch_size, *patch_shape, 1))
    fake = tf.zeros((batch_size, *patch_shape, 1))

    yhat0 = gen(x0, training=False)
    yhat = gen(x, training=False)

    dis0_real = dis(tf.concat([x0, y], axis=-1), training=False)
    dis0_fake = dis(tf.concat([x0, yhat0], axis=-1), training=False)
    dis0_real_loss = dis_loss_fun(real, dis0_real)
    dis0_fake_loss = dis_loss_fun(fake, dis0_fake)

    dis_real = dis(tf.concat([x, y], axis=-1), training=False)
    dis_fake = dis(tf.concat([x, yhat], axis=-1), training=False)
    dis_real_loss = dis_loss_fun(real, dis_real)
    dis_fake_loss = dis_loss_fun(fake, dis_fake)

    gen0_adv_loss = dis_loss_fun(real, dis0_fake)
    gen0_img_loss = img_loss_fun(y, yhat0)
    gen0_loss = gen0_adv_loss + img_loss_weight * gen0_img_loss

    gen_adv_loss = dis_loss_fun(real, dis_fake)
    gen_img_loss = img_loss_fun(y, yhat)
    gen_loss = gen_adv_loss + img_loss_weight * gen_img_loss


    return (dis0_real_loss, dis_real_loss, dis0_fake_loss, dis_fake_loss,
            gen0_loss, gen_loss, gen0_img_loss, gen_img_loss)



@tf.function
def val_trans_step(x0, x, y, 
                   gen, dis=None, img_loss_weight=None):
    
    img_loss_fun = gen.loss
    
    yhat0 = gen(x0, training=False)
    img0_loss = img_loss_fun(y, yhat0)

    yhat = gen(x, training=False)
    img_loss = img_loss_fun(y, yhat)        

    return (img0_loss, img_loss)

    
    
@tf.function
def train_corr_step(x0, x, 
                    translator, registrator, reg_loss_weight=None,
                    deformable=False):
    
    opt = registrator.optimizer
    if deformable: 
        img_loss_fun, reg_loss_fun = registrator.loss
    else:
        img_loss_fun = registrator.loss[0]
    
    with tf.GradientTape() as tape:

        x0 = translator(x0, training=True)
        x = translator(x, training=True)
        
        if deformable:
            xhat, transfo = registrator([x0, x], training=True)
        else:
            xhat = registrator([x0, x], training=True)
        
        img_loss = img_loss_fun(x0, xhat)
        loss = img_loss
        if deformable:
            reg_loss = reg_loss_fun(transfo)
            loss = loss + reg_loss_weight * reg_loss
        
    grads = tape.gradient(loss, registrator.trainable_variables)
    opt.apply_gradients(zip(grads, registrator.trainable_variables))
    
    if deformable:
        return img_loss, reg_loss
    else:
        return img_loss
    
    
@tf.function
def val_corr_step(x0, x, 
                  translator, registrator, reg_loss_weight=None,
                  deformable=False):
    
    if deformable: 
        img_loss_fun, reg_loss_fun = registrator.loss
    else:
        img_loss_fun = registrator.loss[0]


    x0 = translator(x0, training=False)
    x = translator(x, training=False)
    
    if deformable:
        xhat, transfo = registrator([x0, x], training=False)
    else:
        xhat = registrator([x0, x], training=False)
    
    img_loss = img_loss_fun(x0, xhat)
    loss = img_loss
    if deformable:
        reg_loss = reg_loss_fun(transfo)
        loss = loss + reg_loss_weight * reg_loss
    
    if deformable:
        return img_loss, reg_loss
    else:
        return img_loss
    
    
    