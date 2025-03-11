#!/bin/bash
dataset_path=/data-ssd/data3/SemanticKITTI_Test/
prediction_path=/home/cooper/data-ssd/data5/cxm/MOS/BEV-MF-PR/prediction_save_dir_KITTI/sequences-81.56-0.257.zip
echo dataset_path '->' $dataset_path
echo prediction_path '->' $prediction_path
python validate_submission.py --task segmentation \
                        $prediction_path \
                        $dataset_path

## bash check_sub.sh
