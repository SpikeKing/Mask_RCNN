#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/6/13
"""

# Load COCO dataset
import os

from root_dir import ROOT_DIR
from samples.coco.coco import CocoDataset

COCO_DIR = os.path.join(ROOT_DIR, 'coco')

dataset = CocoDataset()
dataset.load_coco(COCO_DIR, "train", auto_download=True)
dataset.prepare()

# Print class names
print(dataset.class_names)
