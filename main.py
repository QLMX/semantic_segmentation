#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-17 下午6:34
"""
from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import time
import os, sys
import random

from config import cfg
from network import NetWork

sys.path.append("utils")
from dataset import get_label_info

def train():
    # Specify operating environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    names_list, label_values, name_string = get_label_info(os.path.join(cfg.data_dir, 'class_dict.csv'))

    fashionAI = NetWork(lr=cfg.lr,
                        model=cfg.model,
                        height=cfg.height,
                        width=cfg.width,
                        label_values=label_values,
                        name_string=name_string,
                        name_list=names_list,
                        class_balancing=cfg.class_balancing)

    fashionAI.train(max_epochs=cfg.num_epochs,
                    batch_size=cfg.batch_size,
                    write_summary=cfg.write_summary,
                    freq_summary=cfg.freq_summary,
                    train_dir=os.path.join(cfg.data_dir, 'train.txt'),
                    val_dir=os.path.join(cfg.data_dir, 'val.txt'),
                    model_dir=cfg.model_dir)

if __name__ == "__main__":
    train()