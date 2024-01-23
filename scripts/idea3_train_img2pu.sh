python 2_train_img2pu_new_idea3.py \
	--data_dir_imgs data/shapenet/ShapeNetRendering \
	--data_dir_pcl data/shapenet/ShapeNet_pointclouds \
	--exp expts/4096/vessel_up \
	--gpu 0 \
	--ae_logs expts/4096/vessel \
	--category vessel \
	--bottleneck 512 \
	--up_ratio 4 \
	--loss cd_emd \
	--batch_size 32 \
	--lr 5e-5 \
	--bn_decoder \
	--load_best_ae \
	--max_epoch 20 \
	--print_n 100
	# --sanity_check
