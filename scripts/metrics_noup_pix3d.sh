#!/bin/bash

gpu=0
exp=expts/noup/2/chair
eval_set=test
dataset=pix3d
data_dir_imgs=data/pix3d
data_dir_pcl=data/pix3d/pix3d_pointclouds
declare -a categs=("chair")

for cat in "${categs[@]}"; do
	python metrics_noup.py \
		--gpu $gpu \
		--dataset $dataset \
		--data_dir_imgs ${data_dir_imgs} \
		--data_dir_pcl ${data_dir_pcl} \
		--exp $exp \
		--category $cat \
		--load_best \
		--bottleneck 512 \
		--bn_decoder \
		--eval_set ${eval_set} \
		--batch_size 1
#		--mode lm \
done

declare -a categs=("chair")
for cat in "${categs[@]}"; do
	echo ${cat}
	cat ${exp}/metrics/${eval_set}/${cat}.csv
	echo
done