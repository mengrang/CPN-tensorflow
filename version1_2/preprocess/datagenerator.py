# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
from PIL import Image, ImageEnhance, ImageFilter

# from random_eraser import *


class DataGenerator():
    """
    To process images and labels
    """
    def __init__(self, dress_type, joints_list, img_dir, train_data_file):
        """Initializer
            Args:
            dress_tpye          : Tpye of dress
            joints_name			: List of joints condsidered
            img_dir				: Directory containing every images
            train_data_file		: Text file with training set data

        """
        self.joints_list = joints_list
        self.img_dir = img_dir
        self.train_data_file = train_data_file
        self.dress_type = dress_type
        self.letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

    # --------------------Generator Initialization Methods ---------------------

    def _read_train_data(self):
        """
        To read labels in csv
        """
        self.train_table = []     # The names of images being trained
        self.data_dict = {}       # The labels of images
        label_file = pd.read_csv( self.train_data_file)
        print('READING LABELS OF TRAIN DATA')
        label_file = label_file[label_file.image_category == self.dress_type]  # Only take the type we want
        for i in range(label_file.shape[0]):
            joints = []
            name = str(label_file.at[i, 'image_id'])
            weight = []
            box = []
            for joint_name in self.joints_list:
                joint_value = []
                value = str(label_file.at[i, joint_name])
                value = value.split('_')
                # print(value)
                joint_value.append(int(value[0]))
                joint_value.append(int(value[1]))
                joints.append(joint_value)
                if value[2] != '1':
                    weight.append(0)
                else:
                    weight.append(1)
            # box of body,[x_box_min,y_box_min,x_box_max,y_box_max]
            box.append(self._min_point(joints, 0))
            box.append(self._min_point(joints, 1))
            box.append(max([x[0] for x in joints]))
            box.append(max([x[1] for x in joints]))
            # print(box)
            # print(name)
            joints = np.reshape(joints, (-1, 2))
            self.data_dict[name] = {'box': box, 'joints': joints, 'weights': weight}
            self.train_table.append(name)
        print('LABEL READING FINISHED')
        return [self.train_table, self.data_dict]

    """
    Get the least number of a column of the dataFrame 
    由于现在存在不在图中的关键点，所以box的确定方面还有点问题，直接用能看到的关键点确定box会不会使训练的模型准确度降低？
    """
    def _min_point(self, joints, n):
        min_point = 600
        for joint in joints:
            temp_point = joint[n]
            if 0 < temp_point < min_point:
                min_point = temp_point
        return min_point

    def _randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.train_table)

    def _complete_sample(self, name):
        """ Check if a sample has no missing value
        Args:
            name 	: Name of the sample
        """
        for i in range(self.data_dict[name]['joints'].shape[0]):
            if np.array_equal(self.data_dict[name]['joints'][i], [-1, -1]):
                return False
        return True

    def _give_batch_name(self, batch_size=16, set='train'):
        """ Returns a List of Samples
        Args:
            batch_size	: Number of sample wanted
            set			: Set to use (valid/train)
        """
        list_file = []
        for i in range(batch_size):
            if set == 'train':
                list_file.append(random.choice(self.train_set))
            elif set == 'valid':
                list_file.append(random.choice(self.valid_set))
            else:
                print('Set must be : train/valid')
                break
        return list_file

    def _create_sets(self, validation_rate=0.1):
        """ Select Elements to feed training and validation set
        Args:
            validation_rate		: Percentage of validation data (in ]0,1[, don't waste time use 0.1)
        """
        self.train_dict = {}
        self.valid_dict = {}
        sample = len(self.train_table)
        valid_sample = int(sample * validation_rate)
        self.train_set = self.train_table[:sample - valid_sample]
        self.valid_set = []
        preset = self.train_table[sample - valid_sample:]
        print('START SET CREATION')
        for elem in preset:
            if self._complete_sample(elem):
                self.valid_set.append(elem)
            else:
                self.train_set.append(elem)
        print('SET CREATED')
        np.save('Dataset-Validation-Set', self.valid_set)
        np.save('Dataset-Training-Set', self.train_set)
        print('--Training set :', len(self.train_set), ' samples.')
        print('--Validation set :', len(self.valid_set), ' samples.')

        for item in range(len(self.train_set)):
            self.train_dict[self.train_set[item]] = self.data_dict[self.train_set[item]]

        for item in range(len(self.valid_set)):
            self.valid_dict[self.valid_set[item]] = self.data_dict[self.valid_set[item]]

        trainset_saver = open('trainset.txt','w')
        trainset_saver.write(str(self.train_set))
        trainset_saver.close()
        traindict_saver = open('traindict.txt','w')
        traindict_saver.write(str(self.train_dict))
        traindict_saver.close()
        validset_saver = open('validset.txt','w')
        validset_saver.write(str(self.valid_set))
        validset_saver.close()
        validdict_saver = open('validdict.txt', 'w')
        validdict_saver.write(str(self.valid_dict))
        validdict_saver.close()

    def generate_set(self, rand=False, validationRate=0.1):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._read_train_data()
        if rand:
            self._randomize()
        self._create_sets(validation_rate=validationRate)

    # ---------------------------- Generating Methods --------------------------

    def _make_gaussian(self, height, width, sigma=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]  # 把一行数变成一列数
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def _generate_hm(self, height, width, joints, maxlength, weight):
        """ Generate a full Heap Map for every joints in an array
        Args:
            height			: Wanted Height for the Heat Map
            width			: Wanted Width for the Heat Map
            joints			: Array of Joints
            maxlength		: Length of the Bounding Box
        """
        num_joints = joints.shape[0]
        hm = np.zeros((height, width, num_joints), dtype=np.float32)
        for i in range(num_joints):
            if not (np.array_equal(joints[i], [-1, -1])) and weight[i] == 1:
                s = int(np.sqrt(maxlength) * maxlength * 10 / 4096) + 2
                hm[:, :, i] = self._make_gaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
            else:
                hm[:, :, i] = np.zeros((height, width))
        return hm

    def _crop_data(self, height, width, box, boxp=0.05):
        """ Automatically returns a padding vector and a bounding box given
        the size of the image and a list of joints.
        Args:
            height		: Original Height
            width		: Original Width
            box			: Bounding Box
            joints		: Array of joints
            boxp		: Box percentage (Use 20% to get a good bounding box)
        """
        padding = [[0, 0], [0, 0], [0, 0]]
        # 把裁剪窗口按照人的box向外扩展20%
        crop_box = [box[0] - int(boxp * (box[2] - box[0])), box[1] - int(boxp * (box[3] - box[1])),
                    box[2] + int(boxp * (box[2] - box[0])), box[3] + int(boxp * (box[3] - box[1]))]
        if crop_box[0] < 0: crop_box[0] = 0
        if crop_box[1] < 0: crop_box[1] = 0
        if crop_box[2] > width - 1: crop_box[2] = width - 1
        if crop_box[3] > height - 1: crop_box[3] = height - 1
        new_h = int(crop_box[3] - crop_box[1])
        new_w = int(crop_box[2] - crop_box[0])
        crop_box = [crop_box[0] + new_w // 2, crop_box[1] + new_h // 2, new_w, new_h]
        if new_h > new_w:
            bounds = (crop_box[0] - new_h // 2, crop_box[0] + new_h // 2)
            if bounds[0] < 0:
                padding[1][0] = abs(bounds[0])
            if bounds[1] > width - 1:
                padding[1][1] = abs(width - bounds[1])
        elif new_h < new_w:
            bounds = (crop_box[1] - new_w // 2, crop_box[1] + new_w // 2)
            if bounds[0] < 0:
                padding[0][0] = abs(bounds[0])
            if bounds[1] > height - 1:
                padding[0][1] = abs(height - bounds[1])
        crop_box[0] += padding[1][0]
        crop_box[1] += padding[0][0]
        # padding[0]为高度（y）方向的padding，padding[1]为宽度（x）方向的padding
        return padding, crop_box

    def _crop_data_new(self, height, width):
        """ Automatically returns a padding vector
            Args:
                height		: Original Height
                width		: Original Width
                crop_box    : the center point and final point of crop box
        """
        padding = [[0, 0], [0, 0], [0, 0]]
        crop_box = [width//2, width//2, width-2, width-2]
        if width == height:
            pass
        elif width > height:
            pad_size = (width - height) // 2
            padding[0][0] = padding[0][1] = pad_size
        else:
            pad_size = (height - width) // 2
            padding[1][0] = padding[1][1] = pad_size
            crop_box[2] = crop_box[3] = height-2
            crop_box[1] = crop_box[0] = height//2
        return padding, crop_box

    def _crop_img(self, img, padding, crop_box):
        """ Given a bounding box and padding values return cropped image
        Args:
            img			: Source Image
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode='constant')
        max_length = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_length // 2:crop_box[1] + max_length // 2,
                crop_box[0] - max_length // 2:crop_box[0] + max_length // 2]
        return img

    def _crop(self, img, hm, padding, crop_box):
        """ Given a bounding box and padding values return cropped image and heatmap
        Args:
            img			: Source Image
            hm			: Source Heat Map
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode='constant')
        hm = np.pad(hm, padding, mode='constant')
        max_length = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_length // 2:crop_box[1] + max_length // 2,
              crop_box[0] - max_length // 2:crop_box[0] + max_length // 2]
        hm = hm[crop_box[1] - max_length // 2:crop_box[1] + max_length // 2,
             crop_box[0] - max_length // 2:crop_box[0] + max_length // 2]
        return img, hm

    def _relative_joints(self, box, padding, joints, to_size=32):
        """ Convert Absolute joint coordinates to crop box relative joint coordinates
        (Used to compute Heat Maps)
        Args:
        box			: Bounding Box
            padding	: Padding Added to the original Image
            to_size	: Heat Map wanted Size
        """
        new_j = np.copy(joints)
        max_l = max(box[2], box[3])
        new_j = new_j + [padding[1][0], padding[0][0]]
        new_j = new_j - [box[0] - max_l // 2, box[1] - max_l // 2]
        new_j = new_j * to_size / (max_l + 0.0000001)
        return new_j.astype(np.int32)

    def _rotate_augment(self, img, hm, max_rotation=30):
        """ # TODO : IMPLEMENT DATA AUGMENTATION
        """
        if random.choice([0, 1]):
            r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            img = transform.rotate(img, r_angle, preserve_range=True)
            hm = transform.rotate(hm, r_angle)
        return img, hm

    '''
    def _size_augment(self, img, hm, weight, max_compress_ratio=0.5, img_tosize=256, hm_tosize=32):
        if random.choice([0,1]):
            compress_ratio = np.random.uniform(max_compress_ratio, 1)
            size = compress_ratio * img.shape[0]
            size = round(size)
            # img resize, lower or equal than img size 256
            resized_img = cv2.resize(img, (size,size))
            resized_hm = cv2.resize(hm,(size,size))

            # resized img padding to 256
            resized_img_shape = resized_img.shape
            img_x = resized_img_shape[0]
            img_y = resized_img_shape[1]
            img2 = np.zeros((256, 256, 3), dtype=np.float32)
            img_x_padding = (256 - img_x) // 2
            img_y_padding = (256 - img_y) // 2
            img2[img_x_padding:img_x_padding+img_x, img_y_padding:img_y_padding+img_y, :] = resized_img[:, :, :]

            # resized hm padding to 256
            resized_hm_shape = resized_hm.shape
            hm_x = resized_hm_shape[0]
            hm_y = resized_hm_shape[1]
            hm2 = np.zeros((256, 256, 13), dtype=np.float32)
            hm_x_padding = (256-hm_x)//2
            hm_y_padding = (256-hm_y)//2
            hm2[hm_x_padding:hm_x_padding+hm_x, hm_y_padding:hm_y_padding+hm_y,:] = resized_hm[:, :, :]

            # resized img and hm to aimed size
            img = cv2.resize(img2, (img_tosize, img_tosize))
            hm = cv2.resize(hm2, (hm_tosize, hm_tosize))

            
            joint_coord_set = np.zeros((13, 2))
            for joint_num in range(13):
                hm2[:, :, joint_num] *= (255 / np.max(hm2[:, :, joint_num]))
                if np.min(hm2[:, :, joint_num]) > -10:
                    joint_coord = np.unravel_index(np.argmax(hm2[:, :, joint_num]),
                                                   (32, 32))
                    joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]
                    # print(joint_coord_set[joint_num, :])

            hm = self._generate_hm(32, 32, joint_coord_set, 32, weight)
            
        return img, hm
    '''

    # input image size=512,out image size=512,input joints size = 512
    def _size_augment(self, img, joints, max_compress_ratio=0.5):
        aug_joints = np.copy(joints)
        compress_ratio = np.random.uniform(max_compress_ratio, 1)
        size = compress_ratio * img.shape[0]
        size = round(size)

        # img resize
        resized_img = cv2.resize(img, (size, size))

        # resized img padding to 256
        resized_img_shape = resized_img.shape
        img_x = resized_img_shape[0]
        img_y = resized_img_shape[1]
        img2 = np.zeros((512, 512, 3), dtype=np.float32)
        img_x_padding = (512 - img_x) // 2
        img_y_padding = (512 - img_y) // 2
        img2[img_x_padding:img_x_padding + img_x, img_y_padding:img_y_padding + img_y, :] = resized_img[:, :, :]
        aug_img = img2

        # calculate relative joints after size augmentation
        compress_coord = joints * compress_ratio  # refer to compressed img joints coordinate
        hm_x_padding = img_x_padding
        hm_y_padding = img_y_padding
        aug_joints[:, 1] = compress_coord[:, 1] + hm_x_padding   # padding joints to 256
        aug_joints[:, 0] = compress_coord[:, 0] + hm_y_padding
        aug_joints = aug_joints * 64 / 512  # resize to 32
        aug_joints = aug_joints.astype(np.int32)

        return aug_img, aug_joints

    def _color_augment(self,img,max_brightness_rate=2.0,max_color_rate=2.0,max_contrast_rate=2.0,max_sharpness_rate=3.0):
        image = Image.fromarray(img)
        # image.show()
        # 亮度增强
        if random.choice([0, 1]):
            enh_bri = ImageEnhance.Brightness(image)
            brightness = random.choice([0.5,0.8,1.2,1.5,1.8])
            image = enh_bri.enhance(brightness)
            # image.show()

        # 色度增强
        if random.choice([0, 1]):
            enh_col = ImageEnhance.Color(image)
            color = random.choice([0.5,0.8,1.2,1.5,1.8])
            image = enh_col.enhance(color)
            # image.show()

        # 对比度增强
        if random.choice([0, 1]):
            enh_con = ImageEnhance.Contrast(image)
            contrast = random.choice([0.5,0.8,1.2,1.5,1.8])
            image = enh_con.enhance(contrast)
            # image.show()

        # 锐度增强
        if random.choice([0, 1]):
            enh_sha = ImageEnhance.Sharpness(image)
            sharpness = random.choice([0.5,0.8,1.2,1.5,1.8])
            image = enh_sha.enhance(sharpness)
            # image.show()

        # mo hu
        if random.choice([0, 1]):
            image = image.filter(ImageFilter.BLUR)

        img = np.asarray(image)
        return img

    def _process_train_img_to_center512(self,img,joints):
        img_shape = img.shape
        img_x = img_shape[0]
        img_y = img_shape[1]
        img512 = np.zeros((512, 512, 3), dtype=np.float32)
        img_x_padding = (512 - img_x) // 2
        img_y_padding = (512 - img_y) // 2
        img512[img_x_padding:img_x_padding + img_x, img_y_padding:img_y_padding + img_y, :] = img[:, :, :]

        joints512 = np.copy(joints)
        joints512[:, 1] = joints[:, 1]+img_x_padding
        joints512[:, 0] = joints[:, 0]+img_y_padding
        return img512, joints512

    def _random_erase(self, img, maxlength, num_joints, joints, train_weight):
        erased_img = np.copy(img)

        # calculate gauss sigma
        s = int(np.sqrt(maxlength) * maxlength * 10 / 4096) + 2

        if random.choice([0, 1]):
            erase_index = random.choice(np.arange(num_joints))
            if not (np.array_equal(joints[erase_index], [-1, -1])) and train_weight[erase_index] == 1:
                # print('erase index: ', erase_index)
                # print('erase joints: ', joints[erase_index])

                # erase hm
                # print('before train weight: ', train_weight)
                train_weight[erase_index] = 0
                # print('after train weight: ', train_weight)

                # erase img
                erase_box_top_left = joints[erase_index] - [s // 2, s // 2]
                # print('erase box top left: ', erase_box_top_left)
                erase_box_bottom_right = joints[erase_index] + [s // 2, s // 2]
                # print('erase box bottom right: ', erase_box_bottom_right)

                erased_img[erase_box_top_left[1]:erase_box_bottom_right[1],
                erase_box_top_left[0]:erase_box_bottom_right[0],
                :] = 0

        return erased_img, train_weight

        # ----------------------- Batch Random Generator ----------------------------------
    def _aux_generator(self, batch_size=16, normalize=False, sample_set='train'):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        while True:
            train_img = np.zeros((batch_size, 512, 512, 3), dtype=np.float32)
            train_gtmap = np.zeros((batch_size, 64, 64, len(self.joints_list)), np.float32)
            train_weights = np.zeros((batch_size, len(self.joints_list)), np.float32)
            train_centermap = np.zeros((batch_size, 512, 512, 1), dtype=np.float32)
            i = 0
            while i < batch_size:
                if sample_set == 'train':
                    name = random.choice(self.train_set)
                elif sample_set == 'valid':
                    name = random.choice(self.valid_set)

                # 读关键点信息
                joints = self.data_dict[name]['joints']
                box = self.data_dict[name]['box']
                weight = np.asarray(self.data_dict[name]['weights'])
                # train_weights[i] = weight

                # 读图片
                img = self.open_img(name)

                # color aug
                img = self._color_augment(img)

                # process img to 512
                img, joints = self._process_train_img_to_center512(img, joints)

                # random erase
                img, train_weights[i] = self._random_erase(img, 512, len(self.joints_list), joints, weight)

                # crop box
                padd, cbox = self._crop_data_new(img.shape[0], img.shape[1])

                # 图片裁剪
                img = self._crop_img(img, padd, cbox)  # 先按边框裁剪
                img = img.astype(np.float64)
                img = scm.imresize(img, (512, 512))  # 再放大成512

                # 生成centermap
                center_map_i = self._make_gaussian(height=512, width=512, sigma=150, center=None)
                center_map_i = np.asarray(center_map_i)
                # plt.imshow(center_map_i)
                # plt.show()
                train_centermap[i] = center_map_i.reshape(512, 512, 1)


                # generate heatmap and augment size
                if random.choice([0, 1]) == 1:
                    new_j = self._relative_joints(cbox, padd, joints, to_size=512)
                    img, new_j = self._size_augment(img, new_j, 0.5)
                    hm = self._generate_hm(64, 64, new_j, 64, train_weights[i])
                else:
                    new_j = self._relative_joints(cbox, padd, joints, to_size=64)
                    hm = self._generate_hm(64, 64, new_j, 64, train_weights[i])


                # img, hm = self._size_augment(img, hm, weight, 0.5, 256, 32)

                # rotate augmentation
                img, hm = self._rotate_augment(img, hm)

                if normalize:
                    train_img[i] = img.astype(np.float32) / 255
                else:
                    train_img[i] = img.astype(np.float32)

                train_gtmap[i] = hm
                i = i + 1
            yield train_img, train_gtmap, train_centermap, train_weights

    # generator batch in order, not random
    def order_generator(self, batch_size=16, normalize=False, sample_set='train'):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        if sample_set == 'train':
            epcho_num_train = len(self.train_set)//batch_size
            train_iter = 0
            while train_iter <= epcho_num_train:
                train_img = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
                train_gtmap = np.zeros((batch_size, 32, 32, len(self.joints_list)), np.float32)
                train_weights = np.zeros((batch_size, len(self.joints_list)), np.float32)
                train_centermap = np.zeros((batch_size, 256, 256, 1), dtype=np.float32)
                i = 0
                while i < batch_size:
                    name = self.train_set[train_iter*batch_size + i]

                    # 读关键点信息
                    joints = self.data_dict[name]['joints']
                    box = self.data_dict[name]['box']
                    weight = np.asarray(self.data_dict[name]['weights'])
                    train_weights[i] = weight

                    # 读图片生成hetmap
                    img = self.open_img(name)
                    padd, cbox = self._crop_data_new(img.shape[0], img.shape[1])
                    new_j = self._relative_joints(cbox, padd, joints, to_size=32)
                    hm = self._generate_hm(32, 32, new_j, 32, weight)

                    # 生成centermap
                    center_map_i = self._make_gaussian(height=256, width=256, sigma=3, center=None)
                    center_map_i = np.asarray(center_map_i)
                    # plt.imshow(center_map_i)
                    # plt.show()
                    train_centermap[i] = center_map_i.reshape(256, 256, 1)

                    # 图片裁剪
                    img = self._crop_img(img, padd, cbox)  # 先按边框裁剪
                    img = img.astype(np.float64)
                    img = scm.imresize(img, (256, 256))  # 放大成256
                    if normalize:
                        train_img[i] = img.astype(np.float32) / 255
                    else:
                        train_img[i] = img.astype(np.float32)
                    train_gtmap[i] = hm
                    i = i + 1
                train_iter = train_iter + 1
                if train_iter > epcho_num_train:
                    train_iter = 0
                yield train_img, train_gtmap, train_centermap, train_weights

        elif sample_set == 'valid':
            epcho_num_valid = len(self.valid_set) // batch_size
            valid_iter = 0
            while valid_iter <= epcho_num_valid:
                valid_img = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
                valid_gtmap = np.zeros((batch_size, 32, 32, len(self.joints_list)), np.float32)
                valid_weights = np.zeros((batch_size, len(self.joints_list)), np.float32)
                valid_centermap = np.zeros((batch_size, 256, 256, 1), dtype=np.float32)
                i = 0
                while i < batch_size:
                    name = self.valid_set[valid_iter * batch_size + i]

                    # 读关键点信息
                    joints = self.data_dict[name]['joints']
                    box = self.data_dict[name]['box']
                    weight = np.asarray(self.data_dict[name]['weights'])
                    valid_weights[i] = weight

                    # 读图片生成hetmap
                    img = self.open_img(name)
                    padd, cbox = self._crop_data_new(img.shape[0], img.shape[1])
                    new_j = self._relative_joints(cbox, padd, joints, to_size=32)
                    hm = self._generate_hm(32, 32, new_j, 32, weight)

                    # 生成centermap
                    center_map_i = self._make_gaussian(height=256, width=256, sigma=3, center=None)
                    center_map_i = np.asarray(center_map_i)
                    # plt.imshow(center_map_i)
                    # plt.show()
                    valid_centermap[i] = center_map_i.reshape(256, 256, 1)

                    # 图片裁剪
                    img = self._crop_img(img, padd, cbox)  # 先按边框裁剪
                    img = img.astype(np.float64)
                    img = scm.imresize(img, (256, 256))  # 再放大成256
                    if normalize:
                        valid_img[i] = img.astype(np.float32) / 255
                    else:
                        valid_img[i] = img.astype(np.float32)
                    valid_gtmap[i] = hm
                    i = i + 1
                valid_iter = valid_iter + 1
                if valid_iter > epcho_num_valid:
                    valid_iter = 0
                yield valid_img, valid_gtmap, valid_centermap, valid_weights

    def generator(self, batchSize=16, norm=True, sample='train'):
        """ Create a Sample Generator
        Args:
            batchSize 	: Number of image per batch
            stacks 	 	: Stacks in HG model
            norm 	 	 	: (bool) True to normalize the batch
            sample 	 	: 'train'/'valid' Default: 'train'
        """
        return self._aux_generator(batch_size=batchSize, normalize=norm, sample_set=sample)

    # ---------------------------- Image Reader --------------------------------
    def open_img(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        if name[-1] in self.letter:
            name = name[:-1]
        img = cv2.imread(os.path.join(self.img_dir, name))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

    def plot_img(self, name, plot='cv2'):
        """ Plot an image
        Args:
            name	: Name of the Sample
            plot	: Library to use (cv2: OpenCV, plt: matplotlib)
        """
        if plot == 'cv2':
            img = self.open_img(name, color='BGR')
            cv2.imshow('Image', img)
        elif plot == 'plt':
            img = self.open_img(name, color='RGB')
            plt.imshow(img)
            plt.show()

    def test(self, toWait=0.2):
        """ TESTING METHOD
        You can run it to see if the preprocessing is well done.
        Wait few seconds for loading, then diaporama appears with image and highlighted joints
        /!\ Use Esc to quit
        Args:
            toWait : In sec, time between pictures
        """
        self._read_train_data()
        self._create_sets()
        for i in range(1):
            img = self.open_img(self.train_set[i])
            w = self.data_dict[self.train_set[i]]['weights']
            padd, box = self._crop_data_new(img.shape[0], img.shape[1])
            new_j = self._relative_joints(box, padd, self.data_dict[self.train_set[i]]['joints'], to_size=256)
            rhm = self._generate_hm(256, 256, new_j, 256, w)
            rimg = self._crop_img(img, padd, box)
            rimg = scm.imresize(rimg, (256, 256))
            grimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
            cv2.imshow('image', grimg / 255 + np.sum(rhm, axis=2))
            # Wait
            time.sleep(toWait)
            if cv2.waitKey(1) == 27:
                print('Ended')
                cv2.destroyAllWindows()
                break

    def test2(self):
        self._read_train_data()
        self._create_sets()

        while True:
            name = 'Images/blouse/0a4cb1d15b17e9deca76c3920fec198d.jpg'

            # 读关键点信息
            joints = self.data_dict[name]['joints']
            box = self.data_dict[name]['box']
            weight = np.asarray(self.data_dict[name]['weights'])

            # 读图片生成box
            img = self.open_img(name)
            plt.imshow(img)
            plt.show()

            # img = self._color_augment(img)

            # process img to 512
            img, joints = self._process_train_img_to_center512(img, joints)
            img, train_weights = self._random_erase(img, 512, len(self.joints_list), joints, weight)
            plt.imshow(img)
            plt.show()

            padd, cbox = self._crop_data_new(img.shape[0], img.shape[1])

            # 图片裁剪
            img = self._crop_img(img, padd, cbox)  # 先按边框裁剪
            img = img.astype(np.float64)
            img = scm.imresize(img, (512, 512))  # 再放大成512

            plt.imshow(img)
            plt.show()

            # generate heatmap and augment size
            if random.choice([1]) == 1:
                new_j = self._relative_joints(cbox, padd, joints, to_size=512)
                img, new_j = self._size_augment(img, new_j, 0.5)
                hm = self._generate_hm(64, 64, new_j, 64, train_weights)
            else:
                new_j = self._relative_joints(cbox, padd, joints, to_size=64)
                hm = self._generate_hm(64, 64, new_j, 64, train_weights)
            plt.imshow(img)
            plt.show()

            # rotate augmentation
            # img, hm = self._rotate_augment(img, hm)

            hm2 = np.amax(hm, axis=2)
            hm2 = cv2.resize(hm2,(512, 512))
            plt.imshow(hm2)
            plt.show()

            grimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imshow('image', grimg / 255 * 0.5 + hm2 * 0.5)
            cv2.waitKey(1000)
