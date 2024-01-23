#!/bin/bash
gpu=0
exp=expts/noup/2/chair
eval_set=valid
dataset=shapenet
data_dir_imgs=data/shapenet/ShapeNetRendering
data_dir_pcl=data/shapenet/ShapeNet_pointclouds
declare -a categs=("chair")

for cat in "${categs[@]}"; do
	echo python metrics_noup.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp  --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24
	python metrics_noup.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp  --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24
done

declare -a categs=("chair")
for cat in "${categs[@]}"; do
	echo ${cat}
	cat ${exp}/metrics/${eval_set}/${cat}.csv
	echo
done
