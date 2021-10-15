_base_ = 'coco_detection.py'

dataset_type = 'HelmetAssignmentDataset'
data_root = 'data/'

data = dict(
    train=dict(
        type='HelmetAssignmentDataset',
        ann_file='data/dataset/labels_3_videos.jsonl',
        img_prefix='data/dataset/train_frames_3_videos/',
        tracking_prefix='data/dataset/train_tracking_3_videos'
        ),
    val=dict(
        type='HelmetAssignmentDataset',
        ann_file='data/dataset/labels_3_videos.jsonl',
        img_prefix='data/dataset/train_frames_3_videos/',
        tracking_prefix='data/dataset/train_tracking_3_videos'
        ),
    test=dict(
        type='HelmetAssignmentDataset',
        ann_file='data/dataset/labels_3_videos.jsonl',
        img_prefix='data/dataset/train_frames_3_videos/',
        tracking_prefix='data/dataset/train_tracking_3_videos'
        ))