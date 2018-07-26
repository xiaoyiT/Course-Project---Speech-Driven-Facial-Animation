from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.z_dim = z_dim
    self.y_dim = y_dim
    
    self.init_pic = tf.placeholder(tf.float32, [self.batch_size,self.input_height, self.input_width, 3], name='ip')

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')
    
    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')


    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    
    self.data = glob(os.path.join("./data",self.dataset_name, self.input_fname_pattern))
    imreadImg = imread(self.data[0])
    if len(imreadImg.shape) >= 3:
        self.c_dim = imread(self.data[0]).shape[-1]
    else:
        self.c_dim = 1

    self.input_y = self.load_landmarks()
    
    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    #self.y = None
    self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z,self.y)
    self.D, self.D_logits   = self.discriminator(inputs,self.y, reuse=False)
    self.sampler            = self.sampler(self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G,self.y, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_files = self.data[0:self.sample_num]
    sample = [
    get_image(sample_file,
              input_height=self.input_height,
              input_width=self.input_width,
              resize_height=self.output_height,
              resize_width=self.output_width,
              crop=self.crop,
              grayscale=self.grayscale) for sample_file in sample_files]
    if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
        sample_inputs = np.array(sample).astype(np.float32)
    
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      self.data = glob(os.path.join( "./data", config.dataset, self.input_fname_pattern))
      batch_idxs = min(int(len(self.data)/self.batch_size), config.train_size)
      
      shift_id = np.arange(len(self.data))
      np.random.shuffle(shift_id)

      for idx in xrange(0, batch_idxs):
        batch_files = []
        batch_y = []
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)
        for i in shift_id[idx*config.batch_size:(idx+1)*config.batch_size]:
          batch_files.append(self.data[i])
          batch_y.append(self.input_y[i])
        batch = [
          get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for batch_file in batch_files]
        if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
            batch_images = np.array(batch).astype(np.float32)
			
        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={ self.inputs: batch_images, self.z: batch_z, self.y: batch_y })
        self.writer.add_summary(summary_str, counter)      
        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={ self.z: batch_z, self.y: batch_y })
        self.writer.add_summary(summary_str, counter)      
        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={ self.z: batch_z, self.y: batch_y })
        self.writer.add_summary(summary_str, counter)        
        errD_fake = self.d_loss_fake.eval({ self.z: batch_z , self.y: batch_y})
        errD_real = self.d_loss_real.eval({ self.inputs: batch_images, self.y: batch_y})
        errG = self.g_loss.eval({self.z: batch_z, self.y: batch_y})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))
        if np.mod(counter, 50) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image,y , reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      
      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      
      h0 = lrelu(conv2d(image, 64, name='d_h0_conv'))
      h1 = tf.nn.max_pool(h0,[1,2,2,1],[1,2,2,1],padding='SAME',name='d_h1_pool')
      h1c = conv_cond_concat(h1, yb)
      h2 = lrelu(self.d_bn1(conv2d(h1c, 128, name='d_h2_conv')))
      h3 = tf.nn.max_pool(h2,[1,2,2,1],[1,2,2,1],padding='SAME',name='d_h3_pool')
      h4 = lrelu(self.d_bn2(conv2d(h3, 256, name='d_h4_conv')))
      h5 = lrelu(self.d_bn3(conv2d(h4, 1024, name='d_h5_conv')))
      h5 = tf.reshape(h5, [self.batch_size, -1])
      h6 = linear(h5, self.z_dim, 'd_h6_lin')
        
      return tf.nn.tanh(h6), h6      



  def generator(self, z, y):
    with tf.variable_scope("generator") as scope:
      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

      self.z_, self.h0_w, self.h0_b = linear(z, 1024*4*4, 'g_h0_lin',with_w=True)     
      
      self.h0 = tf.reshape(self.z_,[-1,4,4,1024])
      h0 = tf.nn.relu(self.g_bn0(self.h0))
       
      h1 = deconv2d(h0,[self.batch_size,8,8,256],name='g_h1_dconv')
      h1 = tf.nn.relu(self.g_bn1(h1))
        
      h2 = deconv2d(h1,[self.batch_size,16,16,128],name='g_h2_dconv')
      h2 = tf.nn.relu(self.g_bn2(h2))
        
      h3 = unpool(h2,name='g_h3_unpool')
        
      h4 = deconv2d(h3,[self.batch_size,64,64,64],name='g_h4_dconv')
      h4 = tf.nn.relu(self.g_bn3(h4))
      print(h4.get_shape())
      h4c = conv_cond_concat(h4, yb)
      print(h4c.get_shape())  
      h5 = unpool(h4c,name='g_h5_unpool')
      print(h5.get_shape())  
      h6 = deconv2d(h5,[self.batch_size,256,256,3],name='g_h6_dconv')
      print(h6.get_shape())  
      return tf.nn.tanh(h6)

  def sampler(self, y):
    z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
    _, z = self.discriminator(self.init_pic, y, True)
        
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      self.z_, self.h0_w, self.h0_b = linear(z, 1024*4*4, 'g_h0_lin',with_w=True)
        
      self.h0 = tf.reshape(self.z_,[-1,4,4,1024])
      h0 = tf.nn.relu(self.g_bn0(self.h0))
        
      h1 = deconv2d(h0,[self.batch_size,8,8,256],name='g_h1_dconv')
      h1 = tf.nn.relu(self.g_bn1(h1))
       
      h2 = deconv2d(h1,[self.batch_size,16,16,128],name='g_h2_dconv')
      h2 = tf.nn.relu(self.g_bn2(h2))
        
      h3 = unpool(h2,name='g_h3_unpool')
        
      h4 = deconv2d(h3,[self.batch_size,64,64,64],name='g_h4_dconv')
      h4 = tf.nn.relu(self.g_bn3(h4))
       
      h4c = conv_cond_concat(h4, yb)
        
      h5 = unpool(h4c,name='g_h5_unpool')
        
      h6 = deconv2d(h5,[self.batch_size,256,256,3],name='g_h6_dconv')
        
      return tf.nn.tanh(h6)



  def load_landmarks(self):         
    z = np.load('./data/landmarks.npy')    
    return z  



  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
