#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-17 下午7:11
"""
import cv2, os
import numpy as np
import sys
import csv
import random

sys.path.append('../')
from config import cfg

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
     Must be in CSV format!
    :param csv_path: The file path of the class dictionairy
    :return: Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
            # print(class_dict)

    class_names_string = ""
    for class_name in class_names:
        if not class_name == class_names[-1]:
            class_names_string = class_names_string + class_name + ", "
        else:
            class_names_string = class_names_string + class_name

    return class_names, label_values, class_names_string

def readData(path):
    img = []
    ann = []
    with open(path, 'r') as f:
        for line in f.readlines():
            item = line.strip().split(',')
            if len(item) == 1:
                img.append(item[0])
            else:
                img.append(item[0])
                ann.append(item[1])
    return img, ann

def prepare_data(dataset_dir=cfg.data_dir):
    """
    Get a list of the training, validation, and testing file paths
    :param dataset_dir:
    :return:
    """
    train_img, train_ann = readData(dataset_dir + 'train.txt')
    val_img, val_ann = readData(dataset_dir + 'val.txt')
    test_img, test_ann = readData(dataset_dir + 'test.txt')
    return train_img, train_ann, val_img, val_ann, test_img, test_ann

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return image

def resizeImage(image, label, size):
    input_image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    output_image = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
    return input_image, output_image

