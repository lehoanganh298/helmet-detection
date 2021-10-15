import sys
sys.path.append('/mnt/d/personal/kaggle/helmet_detection/codes/mmdetection')

import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot



# Deformable DETR
config_file = '/mnt/d/personal/kaggle/helmet_detection/codes/mmdetection/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py'
checkpoint_file = '/mnt/d/personal/kaggle/helmet_detection/codes/mmdetection/checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'

# Deformable DETR two-stages
# config_file = '/mnt/d/personal/kaggle/helmet_detection/codes/mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py'
# checkpoint_file='/mnt/d/personal/kaggle/helmet_detection/codes/mmdetection/checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cpu')
# test a single image
img = '/mnt/d/personal/kaggle/helmet_detection/codes/mmdetection/demo/demo.jpg'
result = inference_detector(model, img)
# show the results
show_result_pyplot(model, img, result)