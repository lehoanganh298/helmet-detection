_base_ = 'coco_detection.py'

dataset_type = 'HelmetAssignmentDataset'
data_root = 'dataset/'

data = dict(
    train=dict(
        type='HelmetAssignmentDataset',
        ann_file='dataset/labels_3_videos.jsonl',
        img_prefix='dataset/train_frames_3_videos/',
        tracking_prefix='dataset/train_tracking_3_videos'
        ),
    val=dict(
        type='HelmetAssignmentDataset',
        ann_file='dataset/labels_3_videos.jsonl',
        img_prefix='dataset/train_frames_3_videos/',
        tracking_prefix='dataset/train_tracking_3_videos'
        ),
    test=dict(
        type='HelmetAssignmentDataset',
        ann_file='dataset/labels_3_videos.jsonl',
        img_prefix='dataset/train_frames_3_videos/',
        tracking_prefix='dataset/train_tracking_3_videos'
        ))