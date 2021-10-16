_base_ = 'coco_detection.py'

dataset_type = 'HelmetDataset'
data_root = 'dataset/'

data = dict(
    train=dict(
        type='HelmetDataset',
        ann_file='dataset/train_dataset.json',
        img_prefix='dataset/images'),
    val=dict(
        type='HelmetDataset',
        ann_file='dataset/val_dataset.json',
        img_prefix='dataset/images'),
    test=dict(
        type='HelmetDataset',
        ann_file='dataset/test_dataset.json',
        img_prefix='dataset/images'))