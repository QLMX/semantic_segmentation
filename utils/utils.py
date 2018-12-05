#!/usr/bin/env python
# encoding: utf-8
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 18-10-18 上午12:34
"""
import numpy as np
import sys, datetime
import os, cv2
import tensorflow as tf

from scipy.misc import imread
from sklearn.metrics import precision_score, recall_score, f1_score

from mergeImage import get_black, image_blur, alpha_merge, edge_process, save_img


def compute_class_weights(image_files, label_values):
    """
    compute all class weights
    :param image_files: Directory where the image segmentation labels are
    :param label_values: the number of classes of pixels in all images
    :return: a list of class weights where each index represents each class label and the element is the class weight for that label.
    """

    # image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]
    num_classes = len(label_values)
    class_pixels = np.zeros(num_classes)

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis=-1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels == 0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights

def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis=-1)
    return x

def tf_reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = tf.argmax(image, axis=-1)
    return x

def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

    # class_dict = get_class_dict("CamVid/class_dict.csv")
    # gt = cv2.imread("CamVid/test_labels/0001TP_007170_L.png",-1)
    # gt = reverse_one_hot(one_hot_it(gt, class_dict))
    # gt = colour_code_segmentation(gt, class_dict)

    # file_name = "gt_test.png"
    # cv2.imwrite(file_name,np.uint8(gt))

def tf_colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies

def compute_mean_iou(pred, label):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    mean_iou = np.mean(I / U)
    return mean_iou

def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

def writer1(output_dir, label_values, queue, stop_token='stop'):

    while True:
        token, path, size, img, output_image = queue.get()
        if token == stop_token:
            return

        img = img[0, :, :, :] * 255

        size = (size[0][0], size[0][1])

        output_single_image = np.array(output_image)
        output_single_image = np.array(output_single_image[0, :, :, :])
        output_image = reverse_one_hot(output_single_image)
        out_vis_image = colour_code_segmentation(output_image, label_values)

        dir = path[0].decode('ascii')
        file_name = filepath_to_name(dir)



        out_vis_image = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
        ori_out_vis = cv2.resize(out_vis_image, size, interpolation=cv2.INTER_NEAREST)

        ori_out_vis[ori_out_vis < 0.5 * 255] = 0
        ori_out_vis[ori_out_vis >= 0.5 * 255] = 255

        save_ori_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR)
        save_ori_img = cv2.resize(save_ori_img, size, interpolation=cv2.INTER_NEAREST)
        # transparent_image = np.append(np.array(save_ori_img)[:, :, 0:3], out_vis_image[:, :, None], axis=-1)
        # transparent_image = Image.fromarray(transparent_image)

        white = save_ori_img.copy()
        white[:, :, 0:3] = 255

        blue = save_ori_img.copy()
        blue[:, :, 0:2] = 0
        blue[:, :, 2] = 255

        red = save_ori_img.copy()
        red[:, :, 0] = 255
        red[:, :, 1:3] = 0

        mask_image = cv2.resize(ori_out_vis, size, interpolation=cv2.INTER_NEAREST)
        mask_image[mask_image < 0.5 * 255] = 0
        mask_image[mask_image >= 0.5 * 255] = 255

        img2gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)  # 与mask颜色相反，白色变成黑色，黑变白
        #
        white_img1_bg = cv2.bitwise_and(white, white, mask=mask_inv)
        white_img2_fg = cv2.bitwise_and(save_ori_img, save_ori_img, mask=mask)
        white_img = cv2.add(white_img1_bg, white_img2_fg)

        blue_mg1_bg = cv2.bitwise_and(blue, blue, mask=mask_inv)
        blue_img2_fg = cv2.bitwise_and(save_ori_img, save_ori_img, mask=mask)
        blue_img = cv2.add(blue_mg1_bg, blue_img2_fg)

        red_mg1_bg = cv2.bitwise_and(red, red, mask=mask_inv)
        red_img2_fg = cv2.bitwise_and(save_ori_img, save_ori_img, mask=mask)
        red_img = cv2.add(red_mg1_bg, red_img2_fg)

        cv2.imwrite(output_dir + "/%s_img.jpg" % (file_name), save_ori_img)
        cv2.imwrite(output_dir + "/%s_ori_pred.png" % (file_name), ori_out_vis)
        cv2.imwrite(output_dir + "/%s_mat_white.jpg" % (file_name), white_img)  #白色
        cv2.imwrite(output_dir + "/%s_mat_blue.jpg" % (file_name), blue_img)  # 白色
        cv2.imwrite(output_dir + "/%s_mat_red.jpg" % (file_name), red_img)  # 白色


def writer(output_dir, label_values, queue, stop_token='stop'):

    while True:
        token, path, size, img, output_image = queue.get()
        if token == stop_token:
            return

        img = img[0, :, :, :] * 255

        size = (size[0][0], size[0][1])

        output_single_image = np.array(output_image)
        output_single_image = np.array(output_single_image[0, :, :, :])
        output_image = reverse_one_hot(output_single_image)
        out_vis_image = colour_code_segmentation(output_image, label_values)

        dir = path[0].decode('ascii')
        file_name = filepath_to_name(dir)



        out_vis_image = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
        mask = cv2.resize(out_vis_image, size, interpolation=cv2.INTER_NEAREST)

        mask[mask < 130] = 0
        mask[mask >= 130] = 255

        image = cv2.imread(dir)

        mask1, left_up, right_down = image_blur(mask, (5, 5), 80)
        ww_img, bb_img, rr_img = get_black(size, 'BGR')

        alpha_img_w = alpha_merge(image, ww_img, mask1)
        alpha_img_b = alpha_merge(image, bb_img, mask1)
        alpha_img_r = alpha_merge(image, rr_img, mask1)

        img_w = edge_process(alpha_img_w)
        img_b = edge_process(alpha_img_b)
        img_r = edge_process(alpha_img_r)

        save_img(img_w, left_up, right_down, '../result/test/w_' + file_name, 1)
        save_img(img_b, left_up, right_down, '../result/test/b_' + file_name, 1)
        save_img(img_r, left_up, right_down, '../result/test/r_' + file_name, 1)


