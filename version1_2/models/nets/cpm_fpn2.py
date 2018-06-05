import tensorflow as tf
import pickle


class CPM_Model(object):
    def __init__(self, input_size, heatmap_size, batch_size, stages, joints, img_type= 'RGB', is_training=True):    # is_trainng not be used
        self.stages = stages
        self.stage_heatmap = []
        self.stage_loss = [0] * stages
        self.total_loss = 0
        self.input_image = None
        self.center_map = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.joints = joints
        self.inference_type = 'Train'  # not be used
        self.batch_size_np = batch_size
        self.stage_loss_batch_hmindex = [0] * self.joints
        self.stage_loss_batch = [0] * self.batch_size_np
        # self.train_weights = tf.zeros((self.batch_size_np, self.joints), dtype=tf.float32)

        if img_type == 'RGB':
            self.input_images = tf.placeholder(dtype=tf.float32,
                                               shape=(None, input_size, input_size, 3),
                                               name='input_placeholder')
        elif img_type == 'GRAY':
            self.input_images = tf.placeholder(dtype=tf.float32,
                                               shape=(None, input_size, input_size, 1),
                                               name='input_placeholder')

        self.cmap_placeholder = tf.placeholder(dtype=tf.float32,
                                               shape=(None, input_size, input_size, 1),
                                               name='cmap_placeholder')
        self.gt_hmap_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(None, heatmap_size, heatmap_size, joints),     # not the same with cpm_hand，hand里头加了1，为何？背景？
                                                  name='gt_hmap_placeholder')
        self.train_weights_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(None,joints),
                                                  name='train_weights_placeholder')
        self._build_model()

    def create_variables(self,shape, name, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
        new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer, regularizer=regularizer)
        return new_variables

    def conv2d_transpose_strided(self, x, W, b, output_shape=None):
        # print x.get_shape()
        # print W.get_shape()
        if output_shape is None:
            output_shape = x.get_shape().as_list()
            output_shape[1] *= 2
            output_shape[2] *= 2
            output_shape[3] = W.get_shape().as_list()[2]
        # print output_shape
        conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding="SAME")
        return tf.nn.bias_add(conv, b)

    def batch_normalization_layer(self,input_layer, dimension):
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.Variable(initial_value=tf.zeros(dimension, dtype=tf.float32))
        gamma = tf.Variable(initial_value=tf.ones(dimension, dtype=tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)

        return bn_layer

    def bn_conv_resblock(self,input_layer, filter_shape, name):
        with tf.variable_scope(name):
            in_channel1 = input_layer.get_shape().as_list()[-1]
            with tf.variable_scope('bn1'):
                bn1_layer = self.batch_normalization_layer(input_layer, in_channel1)
            relu_layer = tf.nn.relu(bn1_layer)

            filter = self.create_variables(shape=filter_shape)
            conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
    
            in_channel2 = conv_layer.get_shape().as_list()[-1]
            with tf.variable_scope('bn2'):
                bn2_layer = self.batch_normalization_layer(conv_layer, in_channel2)
        
            output = bn2_layer + input_layer
            output = tf.nn.relu(output)
        return output

    def bn_conv2_resblock(self,input_layer, filter1_shape, filter2_shape, name):
        with tf.variable_scope(name):
            in_channel1 = input_layer.get_shape().as_list()[-1]
            filter1 = self.create_variables(shape=filter1_shape)
            conv1_layer = tf.nn.conv2d(input_layer, filter1, strides=[1, 1, 1, 1], padding='SAME')
            with tf.variable_scope('bn1'):
                bn1_layer = self.batch_normalization_layer(conv1_layer, in_channel1)
            relu_layer = tf.nn.relu(bn1_layer)

            filter2 = self.create_variables(shape=filter2_shape)
            conv2_layer = tf.nn.conv2d(relu_layer, filter2, strides=[1, 1,1, 1], padding='SAME')
    
            in_channel2 = conv2_layer.get_shape().as_list()[-1]
            with tf.variable_scope('bn2'):
                bn2_layer = self.batch_normalization_layer(conv2_layer, in_channel2)
            
            output = bn2_layer + input_layer
            output = tf.nn.relu(output)
        return output

    def conv1_resblock(self,input_layer, filter_shape, name):
        
        relu_layer = tf.nn.relu(input_layer)
        with tf.variable_scope(name):
            filter = self.create_variables(shape=filter_shape)
            conv2_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, 1, 1, 1], padding='SAME')

            output = conv2_layer + input_layer
            output = tf.nn.relu(output)
        return output

    def conv2_resblock(self,input_layer, filter1_shape,filter2_shape, name):
        with tf.variable_scope(name):
            filter1 = self.create_variables(shape=filter1_shape)
            conv1_layer = tf.nn.conv2d(input_layer, filter1, strides=[1, 1, 1, 1], padding='SAME')
            relu_layer = tf.nn.relu(conv1_layer)

            filter2 = self.create_variables(shape=filter2_shape)
            conv2_layer = tf.nn.conv2d(relu_layer, filter2, strides=[1, 1, 1, 1], padding='SAME')

            output = conv2_layer + input_layer
            output = tf.nn.relu(output)
        return output

    def conv3_resblock(self,input_layer, filter1_shape, filter2_shape, filter3_shape, name):
        with tf.variable_scope(name):
            filter1 = self.create_variables(shape=filter1_shape)
            conv1_layer = tf.nn.conv2d(input_layer, filter1, strides=[1, 1, 1, 1], padding='SAME')
            relu1_layer = tf.nn.relu(conv1_layer)

            filter2 = self.create_variables(shape=filter2_shape)
            conv2_layer = tf.nn.conv2d(relu1_layer, filter2, strides=[1, 1, 1, 1], padding='SAME')
            relu2_layer = tf.nn.relu(conv2_layer)

            filter3 = self.create_variables(shape=filter3_shape)
            conv3_layer = tf.nn.conv2d(relu2_layer, filter3, strides=[1, 1, 1, 1], padding='SAME')

            output = conv3_layer + input_layer
            output = tf.nn.relu(output)
        return output

    def conv4_resblock(self, input_layer, filter1_shape, filter2_shape, filter3_shape, filter4_shape, name):
        with tf.variable_scope(name):
            filter1 = self.create_variables(shape=filter1_shape)
            conv1_layer = tf.nn.conv2d(input_layer, filter1, strides=[1, 1, 1, 1], padding='SAME')
            relu1_layer = tf.nn.relu(conv1_layer)

            filter2 = self.create_variables(shape=filter2_shape)
            conv2_layer = tf.nn.conv2d(relu1_layer, filter2, strides=[1, 1, 1, 1], padding='SAME')
            relu2_layer = tf.nn.relu(conv2_layer)

            filter3 = self.create_variables(shape=filter3_shape)
            conv3_layer = tf.nn.conv2d(relu2_layer, filter3, strides=[1, 1, 1, 1], padding='SAME')
            relu3_layer = tf.nn.relu(conv3_layer)

            filter4 = self.create_variables(shape=filter4_shape)
            conv4_layer = tf.nn.conv2d(relu3_layer, filter4, strides=[1, 1, 1, 1], padding='SAME')

            output = conv4_layer + input_layer
            output = tf.nn.relu(output)
        return output

    def conv6_resblock(self, input_layer, filter1_shape, filter2_shape, filter3_shape, filter4_shape, filter5_shape, filter6_shape, name):
        with tf.variable_scope(name):
            filter1 = self.create_variables(shape=filter1_shape)
            conv1_layer = tf.nn.conv2d(input_layer, filter1, strides=[1, 1, 1, 1], padding='SAME')
            relu1_layer = tf.nn.relu(conv1_layer)

            filter2 = self.create_variables(shape=filter2_shape)
            conv2_layer = tf.nn.conv2d(relu1_layer, filter2, strides=[1, 1, 1, 1], padding='SAME')
            relu2_layer = tf.nn.relu(conv2_layer)

            filter3 = self.create_variables(shape=filter3_shape)
            conv3_layer = tf.nn.conv2d(relu2_layer, filter3, strides=[1, 1, 1, 1], padding='SAME')
            relu3_layer = tf.nn.relu(conv3_layer)

            filter4 = self.create_variables(shape=filter4_shape)
            conv4_layer = tf.nn.conv2d(relu3_layer, filter4, strides=[1, 1, 1, 1], padding='SAME')
            relu4_layer = tf.nn.relu(conv4_layer)

            filter5 = self.create_variables(shape=filter5_shape)
            conv5_layer = tf.nn.conv2d(relu4_layer, filter5, strides=[1, 1, 1, 1], padding='SAME')
            relu5_layer = tf.nn.relu(conv5_layer)

            filter6 = self.create_variables(shape=filter6_shape)
            conv6_layer = tf.nn.conv2d(relu5_layer, filter6, strides=[1, 1, 1, 1], padding='SAME')

            output = conv6_layer + input_layer
            output = tf.nn.relu(output)
        return output

    def _build_model(self):
        with tf.variable_scope('pooled_center_map'):
            self.center_map = tf.layers.average_pooling2d(inputs=self.cmap_placeholder,
                                                          pool_size=[9, 9],
                                                          strides=[8, 8],
                                                          padding='same',
                                                          name='center_map')

        with tf.variable_scope('sub_stages'):
            sub_conv1 = tf.layers.conv2d(inputs=self.input_images,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv1')

            sub_conv2 = tf.layers.conv2d(inputs=sub_conv1,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv2')

            # channel = 64, size = 256
            sub_pool1 = tf.layers.max_pooling2d(inputs=sub_conv2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',  # not the same with cpm_hand，padding方法不一样
                                                name='sub_pool1')

            sub_conv3 = tf.layers.conv2d(inputs=sub_pool1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv3')

            sub_conv4 = tf.layers.conv2d(inputs=sub_conv3,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv4')

            # channel = 128, size = 128
            sub_pool2 = tf.layers.max_pooling2d(inputs=sub_conv4,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',  # not the same with cpm_hand，padding方法不一样
                                                name='sub_pool2')

            sub_conv5 = tf.layers.conv2d(inputs=sub_pool2,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv5')

            sub_conv6 = tf.layers.conv2d(inputs=sub_conv5,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv6')

            sub_conv7 = tf.layers.conv2d(inputs=sub_conv6,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv7')

            sub_conv8 = tf.layers.conv2d(inputs=sub_conv7,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv8')

            # channel = 256, size = 64
            sub_pool3 = tf.layers.max_pooling2d(inputs=sub_conv8,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',  # not the same with cpm_hand，padding方法不一样
                                                name='sub_pool3')

            sub_conv9 = tf.layers.conv2d(inputs=sub_pool3,
                                         filters=512,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv9')

            sub_conv10 = tf.layers.conv2d(inputs=sub_conv9,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv10')

            sub_conv11 = tf.layers.conv2d(inputs=sub_conv10,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv11')

            sub_conv12 = tf.layers.conv2d(inputs=sub_conv11,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv12')

            sub_conv13 = tf.layers.conv2d(inputs=sub_conv12,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv13')

            sub_conv14 = tf.layers.conv2d(inputs=sub_conv13,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv14')

            # # channel = 256, size = 32
            sub_pool4 = tf.layers.conv2d(inputs=sub_conv14,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[2, 2],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_pool4')

            # FPN block
            C5 = sub_pool4  # channel = 256, size = 32
            C4 = sub_pool3  # channel = 256, size = 64
            C3 = sub_pool2  # channel = 128, size = 128

            '''
            C4_conv1 = tf.layers.conv2d(inputs=C4,
                                        filters=256,
                                        kernel_size=[1, 1],
                                        strides=[1, 1],
                                        padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        name='C4_conv1')    # channel = 256, size = 128
            '''
            C4_conv1 = C4

            # 1x1 conv change channels
            C3_conv1 = tf.layers.conv2d(inputs=C3,
                                        filters=256,
                                        kernel_size=[1, 1],
                                        strides=[1, 1],
                                        padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        name='C3_conv1')    # channel = 256, size = 256
            '''
            P5 = tf.layers.conv2d(inputs=C5,
                                  filters=256,
                                  kernel_size=[1, 1],
                                  strides=[1, 1],
                                  padding='same',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name='P5')
            '''
            P5 = C5     # channel = 256,size = 64

            # P5 2x
            W_P5 = self.create_variables(name="W_P5", shape=[3, 3, 256, 256])
            b_P5 = self.create_variables(name="b_P5", shape=[256])
            P5_2x = self.conv2d_transpose_strided(P5, W_P5, b_P5, tf.shape(C4_conv1))     # channel = 256, size = 64

            # P5 4x
            W_P5_4 = self.create_variables(name="W_P5_4", shape=[3, 3, 256, 256])
            b_P5_4 = self.create_variables(name="b_P5_4", shape=[256])
            P5_4x = self.conv2d_transpose_strided(P5_2x, W_P5_4, b_P5_4, tf.shape(C3_conv1))  # channel = 256, size = 128

            P4 = tf.add(C4_conv1, P5_2x, name="P4")      # channel = 256, size = 64
            # P4_2x
            W_P4 = self.create_variables(name="W_P4", shape=[3, 3, 256, 256])
            b_P4 = self.create_variables(name="b_P4", shape=[256])
            P4_2x = self.conv2d_transpose_strided(P4, W_P4, b_P4, tf.shape(C3_conv1))     # channel = 256, size = 128

            # P3
            P3 = tf.add(C3_conv1, P4_2x, name="P3")

            P_cat = tf.concat([P3, P4_2x, P5_4x], axis=3)       # channel = 256, size = 128

            # stride = 2, change size=128 to 64
            sub_conv15 = tf.layers.conv2d(inputs=P_cat,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding='same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv15')      # channel = 256, size = 64

            self.sub_stage_img_feature = tf.layers.conv2d(inputs=sub_conv15,
                                                          filters=128,
                                                          kernel_size=[3, 3],
                                                          strides=[1, 1],
                                                          padding='same',
                                                          activation=tf.nn.relu,
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                          name='sub_stage_img_feature')

        with tf.variable_scope('stage_1'):
            conv1 = tf.layers.conv2d(inputs=self.sub_stage_img_feature,
                                     filters=512,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv1')

            self.stage_heatmap.append(tf.layers.conv2d(inputs=conv1,
                                                       filters=self.joints,     # not the same with cpm_hand，hand里头加了1，为何？背景？
                                                       kernel_size=[1, 1],
                                                       strides=[1, 1],
                                                       padding='same',      # not the same with cpm_hand，padding方法不一样
                                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                       name='stage_heatmap'))

        for stage in range(2, self.stages + 1):
            self._middle_conv(stage)

    def _middle_conv(self, stage):
        with tf.variable_scope('stage_' + str(stage)):
            self.current_featuremap = tf.concat([self.stage_heatmap[stage - 2],
                                                 self.sub_stage_img_feature,
                                                 self.center_map,  # hand这里被注释了，为何？
                                                 ],
                                                axis=3)

            mid_conv1 = tf.layers.conv2d(inputs=self.current_featuremap,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv1')

            mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv2')

            mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv3')

            mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv4')

            mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv5')

            mid_conv6 = tf.layers.conv2d(inputs=mid_conv5,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv6')

            self.current_heatmap = tf.layers.conv2d(inputs=mid_conv6,
                                                    filters=self.joints,  # not the same with cpm_hand，hand里头加了1，为何？背景？
                                                    kernel_size=[1, 1],
                                                    strides=[1, 1],
                                                    padding='same',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='mid_conv7')

            self.stage_heatmap.append(self.current_heatmap)

    # 这里参照cpm_hand做了修改
    def build_loss(self, lr, lr_decay_rate, lr_decay_step, optimizer='Adam'):
        self.gt_heatmap = self.gt_hmap_placeholder
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.optimizer = optimizer
        self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)

        # 计算每个stage的loss
        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        # 计算总loss
        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_step)
            tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer=self.optimizer)
        self.merged_summary = tf.summary.merge_all()

    # new learning rate setting method
    def build_loss2(self, optimizer='Adam'):
        self.gt_heatmap = self.gt_hmap_placeholder
        self.total_loss = 0
        self.optimizer = optimizer
        self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)

        # 计算每个stage的loss
        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        # 计算总loss
        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()

            step = tf.cast(self.global_step, dtype=tf.float64)
            # new learning rate setting method
            epcho = step * 16.0 / 10000
            new_lr = tf.cond(tf.less(epcho, tf.constant(20.0, dtype=tf.float64)),
                     lambda: 0.0006 / 20.0 * epcho,
                     lambda: tf.cond(tf.less(epcho, tf.constant(60.0, dtype=tf.float64)),
                             lambda: tf.constant(0.0006, dtype=tf.float64),
                             lambda: 0.0006 - ((epcho - 60) / (100.0 - 60.0) * 0.0006)
                     ))
            self.lr = new_lr
            tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer=self.optimizer)
        self.merged_summary = tf.summary.merge_all()

    # new learning rate setting method
    def build_loss3(self, optimizer='Adam'):
        self.gt_heatmap = self.gt_hmap_placeholder
        self.train_weights = self.train_weights_placeholder
        self.total_loss = 0
        self.optimizer = optimizer
        self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)

        # 计算每个stage的loss, weighted l2 loss
        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss_batch = [0] * self.batch_size_np
                for batch in range(self.batch_size_np):
                    self.stage_loss_batch_hmindex = [0] * self.joints
                    for hmindex in range(self.joints):
                        self.stage_loss_batch_hmindex[hmindex] = tf.nn.l2_loss(self.stage_heatmap[stage][batch,:,:,hmindex] -
                                                      self.gt_heatmap[batch,:,:,hmindex]) * self.train_weights[batch][hmindex]
                    self.stage_loss_batch[batch] = tf.reduce_sum(self.stage_loss_batch_hmindex)
                self.stage_loss[stage] = tf.reduce_sum(self.stage_loss_batch) / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        # 计算总loss
        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()

            step = tf.cast(self.global_step, dtype=tf.float64)
            # new learning rate setting method
            epcho = step * 8.0 / 10000
            new_lr = tf.cond(tf.less(epcho, tf.constant(20.0, dtype=tf.float64)),
                             lambda: 0.001 / 20.0 * epcho,
                             lambda: tf.cond(tf.less(epcho, tf.constant(40.0, dtype=tf.float64)),
                                             lambda: tf.constant(0.0006, dtype=tf.float64),
                                             lambda: tf.cond(tf.less(epcho, tf.constant(60.0, dtype=tf.float64)),
                                                             lambda: 0.0006 - ((epcho - 40) / (60.0 - 40.0) * 0.0005),
                                                             lambda: tf.constant(0.0001, dtype=tf.float64)
                             )))
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
            ## Pre stage conv
            # conv1
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/bias')

                loaded_kernel = weights['conv1_' + str(layer)]
                loaded_bias = weights['conv1_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv2
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 2) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 2) + '/bias')

                loaded_kernel = weights['conv2_' + str(layer)]
                loaded_bias = weights['conv2_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv3
            for layer in range(1, 5):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 4) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 4) + '/bias')

                loaded_kernel = weights['conv3_' + str(layer)]
                loaded_bias = weights['conv3_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv4
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 8) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 8) + '/bias')

                loaded_kernel = weights['conv4_' + str(layer)]
                loaded_bias = weights['conv4_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv4_CPM
            for layer in range(1, 5):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 10) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 10) + '/bias')

                loaded_kernel = weights['conv4_' + str(2 + layer) + '_CPM']
                loaded_bias = weights['conv4_' + str(2 + layer) + '_CPM_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv5_3_CPM
            conv_kernel = tf.get_variable('sub_stages/sub_stage_img_feature/kernel')
            conv_bias = tf.get_variable('sub_stages/sub_stage_img_feature/bias')

            loaded_kernel = weights['conv4_7_CPM']
            loaded_bias = weights['conv4_7_CPM_b']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            ## stage 1
            conv_kernel = tf.get_variable('stage_1/conv1/kernel')
            conv_bias = tf.get_variable('stage_1/conv1/bias')

            loaded_kernel = weights['conv5_1_CPM']
            loaded_bias = weights['conv5_1_CPM_b']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            if finetune != True:
                conv_kernel = tf.get_variable('stage_1/stage_heatmap/kernel')
                conv_bias = tf.get_variable('stage_1/stage_heatmap/bias')

                loaded_kernel = weights['conv5_2_CPM']
                loaded_bias = weights['conv5_2_CPM_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

                ## stage 2 and behind
                for stage in range(2, self.stages + 1):
                    for layer in range(1, 8):
                        conv_kernel = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/kernel')
                        conv_bias = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/bias')

                        loaded_kernel = weights['Mconv' + str(layer) + '_stage' + str(stage)]
                        loaded_bias = weights['Mconv' + str(layer) + '_stage' + str(stage) + '_b']

                        sess.run(tf.assign(conv_kernel, loaded_kernel))
                        sess.run(tf.assign(conv_bias, loaded_bias))
