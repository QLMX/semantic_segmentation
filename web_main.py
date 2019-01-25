#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2018/12/7 3:24 PM
"""
from __future__ import print_function

import os, sys

from config import cfg
from network import NetWork

sys.path.append("utils")
from dataset import get_label_info

sys.path.append("data")

input_dir = os.path.join(cfg.base_dir, 'Profile/web_server/static/images/upload')

import os
import pyinotify

def model_test(input_img):
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
                        class_balancing=cfg.class_balancing,
                        mode='test')

    fashionAI.test(data_dir=input_img,
                   model_dir=os.path.join(cfg.base_dir, 'checkpoints/checkpoint/FC-DenseNet103/11_19_22/FC-DenseNet103_97'),
                   output_dir=os.path.join(cfg.base_dir, 'Profile/web_server/static/images/download'),
                   threshold=0.5 )

class OnWriteHandler(pyinotify.ProcessEvent):
     def process_IN_CREATE(self, event):
         img_dir = os.path.join(event.path,event.name)
         print("create file: %s " % img_dir) #打印
         model_test(img_dir)


def auto_compile(path='.'):
     wm = pyinotify.WatchManager()
     mask = pyinotify.IN_CREATE #监测类型，如果多种用|分开，pyinotify.IN_CREATE | pyinotify.IN_DELETE
     notifier = pyinotify.Notifier(wm, OnWriteHandler())
     wm.add_watch(path, mask,rec=True,auto_add=True)
     print('==&gt; Start monitoring %s (type c^c to exit)' % path)
     while True:
          try:
               notifier.process_events()
               if notifier.check_events():
                notifier.read_events()
          except KeyboardInterrupt:
               notifier.stop()
               break

if __name__ == "__main__":
     auto_compile(input_dir)