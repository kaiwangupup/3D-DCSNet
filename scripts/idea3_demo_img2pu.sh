#!/bin/bash

gpu=0
exp=expts/4096/telephone_up
eval_set=valid
data_dir_imgs=data/shapenet/ShapeNetRendering
data_dir_pcl=data/shapenet/ShapeNet_pointclouds
cat=telephone

echo python idea3_metrics_img2pu.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24 --visualize
python idea3_metrics_img2pu.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24 --visualize