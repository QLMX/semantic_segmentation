#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-18 上午12:23
"""
import subprocess
import os
import utils
import sys

sys.path.append("..")
sys.path.append("models")
from FC_DenseNet_Tiramisu import build_fc_densenet
from Encoder_Decoder import build_encoder_decoder
from RefineNet import build_refinenet
from FRRN import build_frrn
from MobileUNet import build_mobile_unet
from PSPNet import build_pspnet
from GCN import build_gcn
from DeepLabV3 import build_deeplabv3
from DeepLabV3_plus import build_deeplabv3_plus
from AdapNet import build_adaptnet

def model_checkpoints(url, model_tar, model, model_save):
    subprocess.check_output(['wget', url])
    subprocess.check_output(['tar', '-xvf', model_tar])
    subprocess.check_output(['mv', model, model_save])
    subprocess.check_output(['rm', model_tar])

def download_checkpoints(net, model_name):
    if net == 'VGG':
        if model_name == 'vgg_16':
            model_checkpoints('http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
                              'vgg_16_2016_08_28.tar.gz',
                              'vgg_16.ckpt',
                              'models')
        elif model_name == 'vgg_19':
            model_checkpoints('http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz',
                              'vgg_19_2016_08_28.tar.gz',
                              'vgg_19.ckpt',
                              'models')
    elif net == 'Inception':
        if model_name == 'V1':
            model_checkpoints('http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz',
                              'inception_v1_2016_08_28.tar.gz',
                              'inception_v1.ckpt',
                              'models')
        elif model_name == 'V2':
            model_checkpoints('http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz',
                              'inception_v2_2016_08_28.tar.gz',
                              'inception_v2.ckpt',
                              'models')
        elif model_name == 'V3':
            model_checkpoints('http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
                              'inception_v3_2016_08_28.tar.gz',
                              'inception_v3.ckpt',
                              'models')
        elif model_name == 'V4':
            model_checkpoints('http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
                              'inception_v4_2016_09_09.tar.gz',
                              'inception_v4.ckpt',
                              'models')
        elif model_name == 'resnet_v2':
            model_checkpoints('http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz',
                              'inception_resnet_v2_2016_08_30.tar.gz',
                              'inception_resnet_v2.ckpt',
                              'models')
    elif net == 'ResnetV1':
        if model_name == '50':
            model_checkpoints('http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
                              'resnet_v1_50_2016_08_28.tar.gz',
                              'resnet_v1_50.ckpt',
                              'models')
        elif model_name == '101':
            model_checkpoints('http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz',
                              'resnet_v1_101_2016_08_28.tar.gz',
                              'resnet_v1_101.ckpt',
                              'models')
        elif model_name == '152':
            model_checkpoints('http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz',
                              'resnet_v1_152_2016_08_28.tar.gz',
                              'resnet_v1_152.ckpt',
                              'models')
    elif net == 'ResnetV2':
        if model_name == '50':
            model_checkpoints('http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
                              'resnet_v2_50_2017_04_14.tar.gz',
                              'resnet_v2_50.ckpt',
                              'models')
        elif model_name == '101':
            model_checkpoints('http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
                              'resnet_v2_101_2017_04_14.tar.gz',
                              'resnet_v2_101.ckpt',
                              'models')
        elif model_name == '152':
            model_checkpoints('http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
                              'resnet_v2_152_2017_04_14.tar.gz',
                              'resnet_v2_152.ckpt',
                              'models')


def buildNetwork(model, net_input, num_class):
    # Get the selected model.
    # Some of them require pre-trained ResNet
    if "Res50" in model and not os.path.isfile("models/resnet_v2_50.ckpt"):
        download_checkpoints("ResnetV2", "50")
    if "Res101" in model and not os.path.isfile("models/resnet_v2_101.ckpt"):
        download_checkpoints("ResnetV2", "101")
    if "Res152" in model and not os.path.isfile("models/resnet_v2_152.ckpt"):
        download_checkpoints("ResnetV2", "152")

    network = None
    init_fn = None
    if model == "FC-DenseNet56" or model == "FC-DenseNet67" or model == "FC-DenseNet103":
        network = build_fc_densenet(net_input, preset_model=model, num_classes=num_class)
    elif model == "RefineNet-Res50" or model == "RefineNet-Res101" or model == "RefineNet-Res152":
        # RefineNet requires pre-trained ResNet weights
        network, init_fn = build_refinenet(net_input, preset_model=model, num_classes=num_class)
    elif model == "FRRN-A" or model == "FRRN-B":
        network = build_frrn(net_input, preset_model=model, num_classes=num_class)
    elif model == "Encoder-Decoder" or model == "Encoder-Decoder-Skip":
        network = build_encoder_decoder(net_input, preset_model=model, num_classes=num_class)
    elif model == "MobileUNet" or model == "MobileUNet-Skip":
        network = build_mobile_unet(net_input, preset_model=model, num_classes=num_class)
    elif model == "PSPNet-Res50" or model == "PSPNet-Res101" or model == "PSPNet-Res152":
        # Image size is required for PSPNet
        # PSPNet requires pre-trained ResNet weights
        network, init_fn = build_pspnet(net_input, label_size=[args.crop_height, args.crop_width],
                                        preset_model=model, num_classes=num_class)
    elif model == "GCN-Res50" or model == "GCN-Res101" or model == "GCN-Res152":
        # GCN requires pre-trained ResNet weights
        network, init_fn = build_gcn(net_input, preset_model=model, num_classes=num_class)
    elif model == "DeepLabV3-Res50" or model == "DeepLabV3-Res101" or model == "DeepLabV3-Res152":
        # DeepLabV requires pre-trained ResNet weights
        network, init_fn = build_deeplabv3(net_input, preset_model=model, num_classes=num_class)
    elif model == "DeepLabV3_plus-Res50" or model == "DeepLabV3_plus-Res101" or model == "DeepLabV3_plus-Res152":
        # DeepLabV3+ requires pre-trained ResNet weights
        network, init_fn = build_deeplabv3_plus(net_input, preset_model=model, num_classes=num_class)
    elif model == "AdapNet":
        network = build_adaptnet(net_input, num_classes=num_class)
    elif model == "custom":
        network = build_custom(net_input, num_class)
    else:
        raise ValueError(
            "Error: the model %d is not available. Try checking which models are available using the command python main.py --help")
    return network, init_fn