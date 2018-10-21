#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-18 上午1:22
"""
import cv2, sys
import numpy as np
import tensorflow as tf
import random
from dataset import readData, load_image, resizeImage
from utils import one_hot_it

sys.path.append("..")
from config import cfg

class DataLoader(object):
    """
    A data generator for preprocessing on CPU
    """
    def __init__(self, data_dir='train_path', mode='train', height=384, width=384, label_value=None):
        """
        init
        :param data_dir: str
        :param mode: str, train or test
        """
        self.curr = 0
        self.mode = mode
        self.height = height
        self.width = width
        self.label_value = label_value
        self.img_paths, self.label_paths = readData(data_dir)

        self.n = len(self.img_paths)

    def generator(self, n=0):
        i = 0
        if n == 0:
            n = self.n
        while i < n:
            img_path = self.img_paths[i]
            ann_path = self.label_paths[i]

            img = load_image(img_path)
            ann = load_image(ann_path)
            h, w, _ = img.shape
            size = [w, h]


            ori_img , ann = resizeImage(img, ann, (self.width, self.height))
            # padding

            # img = cv2.copyMakeBorder(img, 0, 512 - h, 0, 512 - w, cv2.BORDER_CONSTANT)
            # heatmap = cv2.resize(heatmap, (512, 512))

            img = ori_img / 255.0

            if self.mode == 'train':
                pass
                #flip image
                if random.random() < cfg.p_flip:
                    img_flip, ann_flip = self.flip(img, ann)
                    ann_flip = np.float32(one_hot_it(label=ann_flip, label_values=self.label_value))
                    yield img_path, size, img_flip, ann_flip

                #rotate image in range of [-30, 30]
                if random.random() < cfg.p_rotate:
                    img_rotate, ann_rotate = self.rotate(img, ann)
                    ann_rotate = np.float32(one_hot_it(label=ann_rotate, label_values=self.label_value))
                    yield img_path, size, img_rotate, ann_rotate

                # hsv image
                if random.random() < cfg.p_hsv:
                    img_hsv = self.random_hsv_transform(ori_img)
                    img_hsv = img_hsv / 255.0
                    ann_hsv = np.float32(one_hot_it(label=ann, label_values=self.label_value))
                    yield img_path, size, img_hsv, ann_hsv

                #gamma
                if random.random() < cfg.p_gamma:
                    img_gamma = self.random_gamma_transform(ori_img)
                    img_gamma = img_gamma / 255.0
                    ann_gamma = np.float32(one_hot_it(label=ann, label_values=self.label_value))
                    yield img_path, size, img_gamma, ann_gamma

            ann_convert = np.float32(one_hot_it(label=ann, label_values=self.label_value))
            yield img_path, size, img, ann_convert

            i += 1

    def rotate(self, img, ann):
        angle = np.random.uniform(-30, 30)
        center = (self.height/2, self.width/2)
        rot_mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
        new_img = cv2.warpAffine(img, rot_mat, (self.height, self.width))
        new_ann = cv2.warpAffine(ann, rot_mat, (self.height, self.width))

        return new_img, new_ann

    def flip(self, img, ann):
        new_img = cv2.flip(img, 1)
        new_ann = cv2.flip(ann, 1)
        return new_img, new_ann

    def _hsv_transform(self, img, hue_delta, sat_mult, val_mult):
        """
        define hsv transformation function
        :param img: original image
        :param hue_delta: Tonal scale
        :param sat_mult: Saturation ratio
        :param val_mult: Proportion of brightness change
        :return:
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        img_hsv[:, :, 1] *= sat_mult
        img_hsv[:, :, 2] *= val_mult
        img_hsv[img_hsv > 255] = 255
        return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2RGB)

    def random_hsv_transform(self, img, hue_vari=10, sat_vari=0.1, val_vari=0.1):
        """
        random transform hsv
        :param img:
        :param hue_vari:
        :param sat_vari:
        :param val_vari:
        :return:
        """
        hue_delta = np.random.randint(-hue_vari, hue_vari)
        sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
        val_mult = 1 + np.random.uniform(-val_vari, val_vari)
        return self._hsv_transform(img, hue_delta, sat_mult, val_mult)

    def _gamma_transform(self, img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table).astype(np.uint8))
        return cv2.LUT(img, gamma_table)


    def random_gamma_transform(self, img, gamma_vari=2.0):
        """
        random gamma transform
        gamma in range of [1/gamma_vari, gamma_vari]
        :param img:
        :param gamma_vari:
        :return:
        """
        log_gamma_vari = np.log(gamma_vari)
        alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
        gamma = np.exp(alpha)
        return self._gamma_transform(img, gamma)

