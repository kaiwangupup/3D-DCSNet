#!/bin/bash

gpu=0
exp=/home/ubuntu/3Dreconstruction/3d-lmnet/expts/6/lm_img2ae_car_emd
#dataset=shapenet

eval_set=valid

image=00.png


python show_results.py \
	--gpu $gpu \
	--exp $exp \
	--mode lm \
	--load_best \
	--bottleneck 512 \
	--bn_decoder \
	--eval_set ${eval_set}\
	--image ${image}\
	--visualize  
