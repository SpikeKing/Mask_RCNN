#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/6/1
"""

import os
import random
import sys

import skimage.io

from root_dir import ROOT_DIR  # 根目录

from samples.coco.coco import CocoConfig  # Coco配置目录，Microsoft的Common Object in Context
from samples.coco.coco import CocoDataset

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

MODEL_DIR = os.path.join(ROOT_DIR, "logs")  # 日志信息和已训练的模型
COCO_DIR = os.path.join(ROOT_DIR, 'coco')

COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'models', "mask_rcnn_coco.h5")  # CoCo的模型

if not os.path.exists(COCO_MODEL_PATH):  # 不存在则下载
    utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_DIR = os.path.join(ROOT_DIR, "images")  # 图片文件夹


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1  # GPU的数量
    IMAGES_PER_GPU = 1  # 每个GPU的图片


config = InferenceConfig()
config.display()  # 输出配置

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)  # 创建推导的模型
model.load_weights(COCO_MODEL_PATH, by_name=True)  # 加载MS-COCO的参数

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

file_names = next(os.walk(IMAGE_DIR))[2]  # 获取图片的文件名
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))  # 随机选择图片

results = model.detect([image], verbose=1)  # 执行检测逻辑

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])  # 数据的可视化


