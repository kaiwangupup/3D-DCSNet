#!/bin/bash

gpu=0
exp=expts/noup/2/sofa
eval_set=valid
data_dir_imgs=data/shapenet/ShapeNetRendering
data_dir_pcl=data/shapenet/ShapeNet_pointclouds
cat=sofa

echo python metrics_noup.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24 --visualize
python metrics_noup.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24 --visualize