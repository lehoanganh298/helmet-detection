import copy
import os.path as osp
import json
import mmcv
import numpy as np
import sys
sys.path.append('/mnt/d/personal/kaggle/helmet_detection/codes/mmdetection')

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module(name='HelmetDataset', force=True)
class HelmetDataset(CustomDataset):

    CLASSES=('Helmet',)
    
    def load_annotations(self, ann_file):
        with open(self.ann_file) as f:
            ann_list = json.load(f)
            
            for item in ann_list:
                bboxes=np.array(item['ann']['bboxes'],dtype=np.float32)
                bboxes[:,2]=bboxes[:,2]+bboxes[:,0]-1
                bboxes[:,3]=bboxes[:,3]+bboxes[:,1]-1
                item['ann']= dict(
#                     bboxes=np.array(item['ann']['bboxes'],dtype=np.float32),
                    bboxes=bboxes,
                    labels=np.array(item['ann']['labels'], dtype=np.long),
                    bboxes_ignore=np.empty(shape=(0,4),dtype=np.float32),
                    labels_ignore=np.array([], dtype=np.long)
                    )

            return ann_list


import os
os.chdir('/mnt/d/personal/kaggle/helmet_detection/')


from mmdet.apis import set_random_seed
from mmcv import Config
cfg = Config.fromfile('codes/mmdetection/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py')

# Modify dataset type and path
cfg.dataset_type = 'HelmetDataset'
cfg.data_root = 'data/'

cfg.data.test.type = 'HelmetDataset'
cfg.data.test.data_root = 'data/'
cfg.data.test.ann_file = 'test_dataset.json'
cfg.data.test.img_prefix = 'images'

cfg.data.train.type = 'HelmetDataset'
cfg.data.train.data_root = 'data/'
cfg.data.train.ann_file = 'train_dataset.json'
cfg.data.train.img_prefix = 'images'

cfg.data.val.type = 'HelmetDataset'
cfg.data.val.data_root = 'data/'
cfg.data.val.ann_file = 'val_dataset.json'
cfg.data.val.img_prefix = 'images'

# modify num classes of the model in box head
cfg.model.bbox_head.num_classes = 1
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# cfg.load_from = 'mmdetection/checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './log_defomable_detr'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 2e-6
# cfg.lr_config_step=1
# cfg.lr_config.warmup = None
# cfg.log_config.interval = 10
cfg.data.samples_per_gpu=1
# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 1

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)


# We can initialize the logger for training and have a look
# at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')

# Build dataset
from mmdet.datasets import build_dataset

datasets = [build_dataset(cfg.data.train)]

from mmdet.models import build_detector

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES


# Train Deformable DETR
from mmdet.apis import train_detector_cpu

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector_cpu(model, datasets, cfg, distributed=False, validate=True)