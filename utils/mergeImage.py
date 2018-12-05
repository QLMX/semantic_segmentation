#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2018/12/5 11:49 PM
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def get_black(size, mode='RGB'):
    white = np.zeros(size, dtype=np.uint8)
    white[:, :, 0:3] = 255

    blue = np.zeros(size, dtype=np.uint8)
    if mode == 'RGB':
        blue[:, :, 0] = 67
        blue[:, :, 1] = 142
        blue[:, :, 2] = 219
    elif mode == 'BGR':
        blue[:, :, 0] = 219
        blue[:, :, 1] = 142
        blue[:, :, 2] = 67

    red = np.zeros(size, dtype=np.uint8)
    if mode == 'RGB':
        red[:, :, 0] = 255
        red[:, :, 1] = 0
        red[:, :, 2] = 0
    elif mode == 'BGR':
        red[:, :, 0] = 0
        red[:, :, 1] = 0
        red[:, :, 2] = 255

    return white, blue, red

def img_show(img, convert=False):
    if convert:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()

def fusion(target, beckground, mask):
    img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)  # 与mask颜色相反，白色变成黑色，黑变白
    #
    img_bg = cv2.bitwise_and(beckground, beckground, mask=mask_inv)
    img_fg = cv2.bitwise_and(target, target, mask=mask)
    img = cv2.add(img_bg, img_fg)
    return img

def alpha_merge(foreground, background, alpha):
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    return outImage.astype(np.uint8)

def image_blur(img, size, threshold):
    img1 = cv2.blur(img, (5, 5))
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.erode(img1, kernel, iterations=1)       #腐蚀图片
    img = cv2.dilate(img, kernel, iterations=1)       #膨胀
    label = img1.copy()
    # is_mask = img > threshold
    # not_mask = img < threshold
    # img[is_mask] = 255
    # img[not_mask] = 0

    label[label == 255] = 0
    label[label == 0] = 0
    label = label[:, :, 0]
    index = np.nonzero(label)
    right_down = np.max(index, axis=1)
    left_up = np.min(index, axis=1)

    return img, left_up, right_down

def crop_size(size, b_size, left_up, right_down):
    rate = min(size[0]/b_size[0], size[1]/b_size[1])
    new_h = int(rate * b_size[1])
    new_w = int(rate * b_size[0])

    person_w = right_down[0] - left_up[0]
    person_h = right_down[1] - left_up[1]

    if new_h < person_h:
        new_h = person_h
    if new_w < person_w:
        new_w = person_w

    add_h = int((new_h - person_h) / 2)

    add_w = int((new_w - person_w) / 2)

    if left_up[0] - add_w < 0:
        x0 = 0
        x1 = right_down[0] + (add_w * 2 - left_up[0])
    elif right_down[0] + add_w > size[0]:
        x0 = left_up[0] - (add_w * 2  - (size[0] - right_down[0]))
        x1 = size[0]
    else:
        x0 = left_up[0] - add_w
        x1 = right_down[0] + add_w

    if left_up[1] - add_h < 0:
        y0 = 0
        y1 = right_down[1] + (add_h * 2 - left_up[1])
    elif right_down[1] + add_h > size[1]:
        y0 = left_up[1] - (add_h * 2  - (size[1] - right_down[1]))
        y1 = size[1]
    else:
        y0 = left_up[1] - add_h
        y1 = right_down[1] + add_h

    return x0, x1, y0, y1

def  crop_size2(size, b_size, left_up, right_down):
    w = right_down[0] - left_up[0]
    add_w = int(w * 0.1)
    w = int(w * 1.1)
    left_up[0] -= add_w
    if left_up[0] < 0:
        left_up[0] = 0
    h = right_down[1] - left_up[1]

    rate = b_size[0] / b_size[1]
    new_w = int(rate * h)

    if new_w > size[0]:
        new_w = size[0]
        new_h = new_w / rate
        sub_h = int((h - new_h) / 2)
        x0 = 0
        x1 = size[0]
        y0 = left_up[1] - sub_h
        y1 = right_down[1] + sub_h
        if y0 < 0:
            y1 -= y0
            y0 = 0
        elif y1 > size[1]:
            y0 -= (y1 - size[1])
            y1 = size[1]

    elif new_w > w and new_w < size[0]:
        new_h = h
        sub_w = int((new_w - w) / 2)
        y0 = left_up[1]
        y1 = right_down[1]
        x0 = left_up[0] - sub_w
        x1 = right_down[0] + sub_w
        if x0 < 0:
            x1 -= x0
            x0 = 0
        elif x1 > size[0]:
            x0 -= (x1 - size[0])
            x1 = size[0]
    else:
        new_w = w
        new_h = w / rate
        sub_h = int((h - new_h) / 2)
        x0 = left_up[0]
        x1 = right_down[0]
        y0 = left_up[1] + sub_h
        y1 = right_down[1] - sub_h

    return x0, x1, y0, y1

def save_img(img, left_up, right_down, path, model):
    if model == 1:
        size = (413, 295)
    elif model == 2:
        size = (626, 413)
    elif model == 5:
        size = (1200, 840)
    elif model == 6:
        size = (1440, 960)
    elif model == 7:
        size = (1920, 1440)
    else:
        print('input model error!')

    w, h, _ = img.shape

    if(w/h < size[0]/size[1]):
        x0, x1, y0, y1 = crop_size((w, h), size, left_up, right_down)
        img = img[x0:x1, y0:y1]

    # x0, x1, y0, y1 = crop_size2((w, h), size, left_up, right_down)
    # img = img[x0:x1, y0:y1]


    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img)

def laplacian_demo(img):
    # 图像梯度，拉普拉斯算子
    # dst = cv.Laplacian(image, cv.CV_32F)

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # 自定义核
    dst = cv2.filter2D(img, cv2.CV_32F, kernel=kernel)
    lpls = cv2.convertScaleAbs(dst)
    return lpls

def edge_process(img):
    img = cv2.blur(img, (7, 7))
    # img = cv2.blur(img, (7, 7))
    # img = cv2.bilateralFilter(img, 0, 100, 15)
    # img = cv2.pyrMeanShiftFiltering(img, 10, 10)
    # img = laplacian_demo(img)
    # img = cv2.cv2.GaussianBlur(img,(5,5),0)
    return img


if __name__=="__main__":
    img_list = os.listdir('img')
    img_list = [i for i in img_list if 'jpg' in i]
    img_list = sorted(img_list)

    ann_list = os.listdir('ann')
    ann_list = [i for i in ann_list if 'png' in i]
    ann_list = sorted(ann_list)


    for i, value in enumerate(img_list):
        img = value
        ann = ann_list[i]

        img1 = cv2.imread('img/' + img)
        mask = cv2.imread('ann/' + ann)
        mask1, left_up, right_down = image_blur(mask, (5,5), 80)
        # mask1 = mask

        size = img1.shape
        ww_img, bb_img, rr_img = get_black(size, 'BGR')

        alpha_img_w = alpha_merge(img1, ww_img, mask1)
        alpha_img_b = alpha_merge(img1, bb_img, mask1)
        alpha_img_r = alpha_merge(img1, rr_img, mask1)

        img_w = edge_process(alpha_img_w)
        img_b = edge_process(alpha_img_b)
        img_r = edge_process(alpha_img_r)


        save_img(img_w, left_up, right_down, 'result/cc/w' + img, 1)
        save_img(img_b, left_up, right_down, 'result/cc/b' + img, 1)
        save_img(img_r, left_up, right_down, 'result/cc/r' + img, 1)