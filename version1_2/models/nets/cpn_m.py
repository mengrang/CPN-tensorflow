# -*- coding: utf-8 -*-
import tensorflow as tf
import pickle
import sys, os
import numpy as np
from functools import partial
import collections
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils
sys.path.append('/home/lzhpc/cpn_m/version1_2/models/nets/')
from cpn_m_utils import activation_summary,batch_normalization_layer,\
    subsample,conv_bn,bottleneck
from config import FLAGS


class CPN_Model(object):

    def __init__(self, input_size, heatmap_size, batch_size, joints, img_type= 'RGB', is_training=True):   
        self.resnet_fms = []
        self.global_fms = []
        self.global_outs = []
        self.refine_fms = []
        self.refine_out = None
        self.global_loss = 0
        self.refine_loss = 0
        self.total_loss = 0
        self.input_image = None
        self.center_map = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.joints = joints
        self.inference_type = 'Train'  # not be used
        self.batch_size_np = batch_size


        if img_type == 'RGB':
            self.input_images = tf.placeholder(dtype=tf.float32,shape=(None, input_size, input_size, 3),
                                                name='input_placeholder')
        elif img_type == 'GRAY':
            self.input_images = tf.placeholder(dtype=tf.float32,shape=(None, input_size, input_size, 1),
                                                name='input_placeholder')

        self.cmap_placeholder = tf.placeholder(dtype=tf.float32,shape=(None, input_size, input_size, 1),name='cmap_placeholder')
        self.gt_hmap_placeholder = tf.placeholder(dtype=tf.float32,shape=(None, heatmap_size, heatmap_size, joints),
                                                name='gt_hmap_placeholder')
        self.train_weights_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,joints),
                                                  name='train_weights_placeholder')
        self._build_model()

    def blocks(self):  
        block = collections.namedtuple('block',['scope','unit_fn','args'])
        blocks = [
            block('block1', bottleneck,[(256, 64, 1)] * 2 + [(256, 64, 1)]),
            block('block2', bottleneck,[(512, 128, 2)] + [(512, 128, 1)] * 3),
            block('block3', bottleneck,[(1024, 256, 2)] + [(1024, 256, 1)] * 22),#22
            block('block4', bottleneck,[(2048, 512, 2)] + [(2048, 512, 1)] * 2)]
        return blocks

    def stack_blocks_dense(self,net,blocks,output_stride=None,outputs_collections=None):
        current_stride = 1
        rate = 1

        for block in blocks:
            with variable_scope.variable_scope(block.scope, 'block', values=[net]) as sc:
                for i, unit in enumerate(block.args):
                    
                    if output_stride is not None and current_stride > output_stride:
                        raise ValueError('The target output_stride cannot be reached.')
                    # with variable_scope.variable_scope('unit_%d' % (i+1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net,
                            depth=unit_depth,
                            depth_bottleneck=unit_depth_bottleneck,
                            stride=1,
                            rate=rate,
                            name='unit_%d' % (i+1))
                        rate *= unit_stride
                    else:
                        net = block.unit_fn(net,
                            depth=unit_depth,
                            depth_bottleneck=unit_depth_bottleneck,
                            stride=unit_stride,
                            rate=1,
                            name='unit_%d' % (i+1))
                        current_stride *= unit_stride
                    
                net = utils.collect_named_outputs(outputs_collections, sc.name, net)
        if output_stride is not None and current_stride != output_stride:
            raise ValueError('The target output_stride cannot be reached.')
        return net

    def _build_model(self):
        blocks = self.blocks()
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('pooled_center_map'):
            self.center_map = tf.layers.average_pooling2d(inputs=self.cmap_placeholder,
                                                          pool_size=[9, 9],
                                                          strides=[8, 8],
                                                          padding='same',
                                                          name='center_map')

        #resnet50                                                  
        with tf.variable_scope('resnet50'):          
            net = bottleneck(inputs=self.input_images,
                                    depth=64,
                                    depth_bottleneck=64,
                                    stride=1,
                                    name='bottleneck1')
            net = subsample(net, factor=2,name='sub1')
            '''
            net = resnet_utils.conv2d_same(
                image, 64, 7, stride=2, scope='conv1')
            '''
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])  #net.get_shape().tolist=[N,H,W,C]
            net = tf.contrib.layers.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
            net = self.stack_blocks_dense(net=net, blocks=blocks[0:1])

            net2 = self.stack_blocks_dense(net=net, blocks=blocks[1:2])
        
            net3 = self.stack_blocks_dense(net=net, blocks=blocks[2:3])
        
            net4 = self.stack_blocks_dense(net=net, blocks=blocks[3:4])

            self.resnet_fms = [net, net2, net3, net4]

        #global_net
        with tf.variable_scope('global_net'):
            
            last_fm = None          
            for i, fm in enumerate(reversed(self.resnet_fms)):     #blocks is the param of the creat_global_net
                with tf.variable_scope('lateral'):
                    '''lateral 1*1conv'''
                    lateral = conv_bn(inputs=fm,
                                    filters=256,
                                    kernel_size=1,
                                    stride=1,
                                    
                                    activation=tf.nn.relu,
                                    
                                    name='res{}'.format(5-i))
                    if last_fm is not None:
                        sz = tf.shape(lateral)
                        upsample = tf.image.resize_bilinear(last_fm, (sz[1], sz[2]), name='upsample/res{}'.format(5-i))# resize the feature map
                        upsample = tf.layers.conv2d(inputs=upsample,
                                                    filters=256, 
                                                    kernel_size=1,
                                                    padding='SAME',
                                                    name='merge/res{}'.format(5-i))
                        last_fm = upsample + lateral
                    else:
                        last_fm = lateral
                        # print(lateral.get_shape())
                with tf.variable_scope('tmp'):
                    tmp = conv_bn(inputs=last_fm,
                                filters=256,
                                kernel_size=1,
                                stride=1,
                                activation=tf.nn.relu, 
                                name='res{}'.format(5-i))
                    
                with tf.variable_scope('pyramid'):
                    out = conv_bn(inputs=tmp,
                                filters=self.joints,
                                kernel_size=3,
                                stride=1,
                                activation=None,
                                name='res{}'.format(5-i))
                    
                self.global_fms.append(last_fm)
                self.global_outs.append(tf.image.resize_bilinear(out, (FLAGS.heatmap_size, FLAGS.heatmap_size)))
            self.global_fms.reverse()
            self.global_outs.reverse()
            
        
        #refien_net
        with tf.variable_scope('refine_net'):
            for i, block in enumerate(self.global_fms):            
                mid_fm = block
                for j in range(i+1):
                    mid_fm = bottleneck(inputs=mid_fm, 
                                        depth=256, 
                                        depth_bottleneck=128, 
                                        stride=1,rate=1,
                                        name='res{}/refine_conv{}'.format(2+i, j)) # no projection
                    with tf.variable_scope('upsample'):
                        mid_fm = tf.image.resize_bilinear(mid_fm, 
                                                        (FLAGS.heatmap_size, FLAGS.heatmap_size),
                                                        name='res{}'.format(2+i))
                self.refine_fms.append(mid_fm)
                
            self.refine_fm = tf.concat(self.refine_fms, axis=3,name='refine_concat')
            self.refine_fm = bottleneck(self.refine_fm,
                                256, 128, 
                                stride=1, 
                                name='final_bottleneck')
            self.refine_out = conv_bn(inputs=self.refine_fm,
                                    filters=self.joints,
                                    kernel_size=3,
                                    stride=1,
                                    activation=None,
                                    name='refine_out')
            print('ro',self.refine_out.get_shape())
    def ohkm(self,loss):
        self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)
        with tf.variable_scope('ohkm_loss'):
            ohkm_loss = 0.
            for i in range(FLAGS.batch_size):
                sub_loss = loss[i]
                topk_val, topk_idx = tf.nn.top_k(sub_loss, 
                                                k=8, 
                                                sorted=False, name='ohkm{}'.format(i))
                tmp_loss = tf.gather(sub_loss, topk_idx, name='ohkm_loss{}'.format(i)) # can be ignore ???
                ohkm_loss += tf.reduce_sum(tmp_loss) / 8
            ohkm_loss /= self.batch_size
        return ohkm_loss    


    def build_loss_ohkm(self, init_lr, lr_decay_rate, lr_decay_step, optimizer='Adam'):
        self.gt_heatmap = self.gt_hmap_placeholder
        self.global_loss = 0
        self.refine_loss = 0
        self.total_loss = 0
        self.learning_rate = init_lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.optimizer = optimizer
        self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)

        global_loss = 0.
        with tf.variable_scope('global_loss'):
            for i, (global_out, global_label) in enumerate(zip(self.global_outs, [self.gt_heatmap]*5)):
                self.global_loss += tf.reduce_mean(tf.square(global_out - global_label)) / len([self.gt_heatmap]*5)
            self.global_loss /= 2.
            tf.summary.scalar('global_loss', self.global_loss)
        with tf.variable_scope('refine_loss'):
            print(self.gt_heatmap.get_shape())
            # self.refine_loss = tf.nn.l2_loss(self.refine_out - self.gt_heatmap,
            #                                            name='l2_loss') / self.batch_size
            self.refine_loss = tf.reduce_mean(tf.square(self.refine_out - self.gt_heatmap), (1,2))                                          
            self.refine_loss = self.ohkm(self.refine_loss)
            tf.summary.scalar('refine_loss', self.refine_loss)
        with tf.variable_scope('total_loss'):
            self.total_loss = self.refine_loss + self.global_loss
            tf.summary.scalar('loss', self.total_loss)
        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()

            step = tf.cast(self.global_step, dtype=tf.float64)
            # new learning rate setting method
            epcho = step * 16.0 / 10000
            new_lr = tf.cond(tf.less(epcho, tf.constant(20.0, dtype=tf.float64)),
                             lambda: 0.001 / 20.0 * epcho,
                             lambda: tf.cond(tf.less(epcho, tf.constant(40.0, dtype=tf.float64)),
                                             lambda: tf.constant(0.0006, dtype=tf.float64),
                                             lambda: 0.0006 - ((epcho - 40) / (60.0 - 40.0) * 0.0006)
                                             ))
            self.lr = new_lr
            tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer=self.optimizer)
        self.merged_summary = tf.summary.merge_all()
    
    def load_weights_from_file(self, weight_file_path, sess, finetune=True):
        weights = pickle.load(open(weight_file_path, 'rb'), encoding='latin1')

        with tf.variable_scope('', reuse=True):
            # resnet50
            for layer in range(1,4):
                conv_kernel = tf.get_variable('resnet50/bottleneck1' + '/conv' +str(layer) +'/kernel')
                conv_bias = tf.get_variable('resnet50/bottleneck1' +'/conv' +str(layer) + '/bias')

                loaded_kernel = weights['resnet50/bottleneck1' + '/conv' +str(layer)]
                loaded_bias = weights['resnet50/bottleneck1' + '/conv' +str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            for layer,block in enumerate(self.blocks):
                for i, unit in enumerate(block.args):
                    for j in range(1,4):
                        conv_kernel = tf.get_variable('resnet50/block' + str(layer+1) + '/unit_' +str(i+1) + '/conv' +str(j)+'/kernel')
                        conv_bias = tf.get_variable('resnet50/block' + str(layer+1) + '/unit_' +str(i+1) + '/conv' +str(j)+ '/bias')

                        loaded_kernel = weights['resnet50/block' + str(layer+1) + '/unit_' +str(i+1) + '/conv' +str(j)]
                        loaded_bias = weights['resnet50/block' + str(layer+1) + '/unit_' +str(i+1) + '/conv' +str(j)+ '_b']

                        sess.run(tf.assign(conv_kernel, loaded_kernel))
                        sess.run(tf.assign(conv_bias, loaded_bias))

            # global_net
            for layer in range(1, 6):
                conv_kernel = tf.get_variable('global_net/lateral/res{}'.format(layer) + '/kernel')
                conv_bias = tf.get_variable('global_net/lateral/res{}'.format(layer) + '/bias')

                loaded_kernel = weights['global_net/lateral/res{}'.format(layer)]
                loaded_bias = weights['global_net/lateral/res{}'.format(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            for layer in range(1, 6):
                conv_kernel = tf.get_variable('global_net/tmp/res{}'.format(layer) + '/kernel')
                conv_bias = tf.get_variable('global_net/tmp/res{}'.format(layer) + '/bias')

                loaded_kernel = weights['global_net/tmp/res{}'.format(layer)]
                loaded_bias = weights['global_net/tmp/res{}'.format(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            for layer in range(1, 6):
                conv_kernel = tf.get_variable('global_net/pyramid/res{}'.format(layer) + '/kernel')
                conv_bias = tf.get_variable('global_net/pyramid/res{}'.format(layer) + '/bias')

                loaded_kernel = weights['global_net/pyramid/res{}'.format(layer)]
                loaded_bias = weights['global_net/pyramid/res{}'.format(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            
            # refine_net
            for layer, block in enumerate(self.global_fms):
                
                for j in range(layer):
                    for i in range(1,4):
                        conv_kernel = tf.get_variable('refien_net/res{}/refien_conv{}'.format(layer+2, j) +'/conv'+str(i+1) +'/kernel')
                        conv_bias = tf.get_variable('refien_net/res{}/refien_conv{}'.format(layer+2, j) +'/conv'+str(i+1)+ '/bias')

                        loaded_kernel = weights['refien_net/res{}/refien_conv{}'.format(layer+2, j) +'/conv'+str(i+1)]
                        loaded_bias = weights['refien_net/res{}/refien_conv{}'.format(layer+2, j) +'/conv'+str(i+1) + '_b']

                        sess.run(tf.assign(conv_kernel, loaded_kernel))
                        sess.run(tf.assign(conv_bias, loaded_bias))
            
            #final_bottleeck
            for layer in range(1,4):
                conv_kernel = tf.get_variable('refien_net/final_bottleneck' +'/conv'+str(i+1) +'/kernel')
                conv_bias = tf.get_variable('refien_net/final_bottleneck' +'/conv'+str(i+1)+ '/bias')

                loaded_kernel = weights['refien_net/final_bottleneck' +'/conv'+str(i+1)]
                loaded_bias = weights['refien_net/final_bottleneck' +'/conv'+str(i+1) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            #refine_out
            for layer in range(1,4):
                conv_kernel = tf.get_variable('refien_net/refine_out' +'/conv'+str(i+1) +'/kernel')
                conv_bias = tf.get_variable('refien_net/refine_out' +'/conv'+str(i+1)+ '/bias')

                loaded_kernel = weights['refien_net/refine_out' +'/conv'+str(i+1)]
                loaded_bias = weights['refien_net/refine_out' +'/conv'+str(i+1) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))



            

    