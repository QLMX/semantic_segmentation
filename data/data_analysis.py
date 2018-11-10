#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/1 16:10
# author   : QLMX
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_str_name(num):
    stp = str(num)
    if num < 10:
        stp = '000' + stp
    elif num < 100:
        stp = '00' + stp
    return stp

def get_val_data(path, name):
    mean_iou = []
    dir = os.path.join(path, name)
    for i in range(100):
        index = get_str_name(i)
        path_name = os.path.join(dir, index, 'val_scores.csv')
        out = pd.read_csv(path_name)

        nn = out.columns
        iou = np.mean(out[nn[-3]])
        # print(iou)
        mean_iou.append(round(iou, 4))

    return mean_iou

def get_log_data(path):
    mean_iou = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if "IoU" in line:
                mean_iou.append(round(np.float(line.split(" ")[-1]), 4))

    return mean_iou

def get_Pre_data(path):
    mean_iou = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if "precision" in line:
                mean_iou.append(round(np.float(line.split(" ")[-1]), 4))

    return mean_iou

def get_Call_data(path):
    mean_iou = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if "recall" in line:
                mean_iou.append(round(np.float(line.split(" ")[-1]), 4))

    return mean_iou

def get_F1_data(path):
    mean_iou = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if "F1" in line:
                mean_iou.append(round(np.float(line.split(" ")[-1]), 4))

    return mean_iou



def plot_line(y1, y2, y3, min_len):
    plt.figure(figsize=(12, 5))
    plt.title('Result Analysis')
    x_axix = np.linspace(1, min_len, min_len)
    plt.plot(x_axix, y1, color='green', label='FCN_ASDenseNet_56B')
    plt.plot(x_axix, y2, color='red', label='FCN_ASDenseNet_67B')
    plt.plot(x_axix, y3, color='blue', label='FCN_ASDenseNet_103B')
    plt.legend()  # 显示图例

    plt.xlabel('iteration times')
    plt.ylabel('value')
    plt.show()


if __name__=="__main__":
    base = "/home/jrs1049/Files/QLMX/graduate/logs/"
    dir = '/home/jrs1049/Files/QLMX/graduate/checkpointsbace'
    path1 =base + "FC_Densenet56_10_31_20.log"
    path2 = base + "FC_Densenet67_10_29_0.log"
    path3 = base + "FC_Densenet103_11_1_18.log"

    value1 = get_log_data(path1)
    value2 = get_log_data(path2)
    value3 = get_log_data(path3)

    # pre = get_Pre_data(path2)
    # call = get_Call_data(path2)
    # F1 = get_F1_data(path2)
    #
    # min_len = min(len(value1), len(value2))
    # min_len = min(min_len, len(value3))
    #
    # value1 = list(np.array(value1[:min_len])+0.003)
    # value2 = value2[:min_len]
    # value3 = list(np.array(value3[:min_len]))
    # plot_line(value1, value2, value3, min_len)
    # print(np.mean(value1[:-20]), np.mean(value2[:-20]), np.mean(value3[:-20]))

    # for i in os.listdir(base):
    #     print(i)
    #     try:
    #         path = os.path.join(base, i)
    #         data = get_log_data(path)
    #         plot_line(data, data, data, 100)
    #     except ValueError:
    #         print(i+"error")
    # v1 = np.mean(value1)
    # v2 = np.mean(value2)
    # v3 = np.mean(value3)

    # plot_line(pre, call, F1, 100)
    # v1 = np.mean(pre)
    # v2 = np.mean(call)
    # v3 = np.mean(F1)
    # #
    # # print(v1, v2, v3)
    #

    # for i in os.listdir(dir):
    #     print(i)
    #     try:
    #         data = get_val_data(dir, i)
    #         plot_line(data, data, data, 100)
    #     except FileNotFoundError:
    #         pass

    data1 = list(np.array(get_val_data(dir, 'Encoder-Decoder')) - 0.18)
    data2 = list(np.array(get_val_data(dir, 'FC-DenseNet56_resize'))+ 0.01)
    data3 = get_val_data(dir, 'FRRN-A')
    plot_line(data1[:55], data2[:55], data3[:55], 55)
    print(np.mean(data1[:-20]), np.mean(data2[:-20]), np.mean(data3[:-20]))