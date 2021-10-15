import json
import numpy as np

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