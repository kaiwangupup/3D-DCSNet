python train_noup.py \
	--data_dir_imgs data/shapenet/ShapeNetRendering \
	--data_dir_pcl data/shapenet/ShapeNet_pointclouds \
	--exp expts/noup/2/table \
	--gpu 0 \
	--category table \
	--bottleneck 512 \
	--loss chamfer \
	--batch_size 32 \
	--lr 5e-5 \
	--bn_decoder \
	--max_epoch 25 \
	--print_n 100
	# --sanity_check
