# -*- coding: utf-8 -*-
class FLAGS(object):

    """
    General settings
    """
    input_size = 512
    heatmap_size = 64
    joint_gaussian_variance = 1.0
    center_radius = 15
    num_of_joints = 15
    color_channel = 'RGB'
    normalize_img = True
    use_gpu = True
    gpu_id = 0
    if_show = False
    img_show_iters = 100

    """
       Training settings
    """
    network_def = 'cpn_m'
    # pretrained_model = './dress/models/weights/cpm_body/input_512_output_64/joints_15/stages_5/init_0.001_rate_0.5_step_100000'
    datagenerator_config_file = './preprocess/config.cfg'
    batch_size = 16
    init_lr = 0.001
    lr_decay_rate = 0.5
    lr_decay_step = 100000
    training_iters = 37501
    verbose_iters = 10
    validation_iters = 100
    model_save_iters = 5000











