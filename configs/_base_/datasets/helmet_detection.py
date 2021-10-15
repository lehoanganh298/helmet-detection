_base_ = 'coco_detection.py'

dataset_type = 'HelmetDataset'
data_root = 'data/'

data = dict(
    train=dict(
        type='HelmetDataset',
        ann_file='data/train_dataset.json',
        img_prefix='data/images'),
    val=dict(
        type='HelmetDataset',
        ann_file='data/val_dataset.json',
        img_prefix='data/images'),
    test=dict(
        type='HelmetDataset',
        ann_file='data/test_dataset.json',
        img_prefix='data/images'))