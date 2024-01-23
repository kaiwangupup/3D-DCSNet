from importer import *
from utils.icp import icp
from tqdm import tqdm
from utils.encoder_2 import *

parser = argparse.ArgumentParser()

# Machine Details
parser.add_argument('--gpu', type=str, required=True, help='[Required] GPU to use')

# Dataset
#parser.add_argument('--dataset', type=str, required=True, help='Choose from [shapenet, pix3d]')
parser.add_argument('--data_dir_imgs', type=str, required=True, help='Path to shapenet rendered images')
parser.add_argument('--data_dir_pcl', type=str, required=True, help='Path to shapenet pointclouds')

# Experiment Details
#parser.add_argument('--mode', type=str, required=True, help='[Required] Latent Matching setup. Choose from [lm, plm]')
parser.add_argument('--exp', type=str, required=True, help='[Required] Path of experiment for loading pre-trained model')
parser.add_argument('--category', type=str, required=True, help='[Required] Model Category for training')
parser.add_argument('--load_best', action='store_true', help='load best val model')

# AE Details
parser.add_argument('--bottleneck', type=int, required=False, default=512, help='latent space size')
parser.add_argument('--up_ratio', type=int, default=4, help='up sampling ratio')
# parser.add_argument('--bn_encoder', action='store_true', help='Supply this parameter if you want bn_encoder, otherwise ignore')
parser.add_argument('--bn_decoder', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')
parser.add_argument('--bn_decoder_final', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')

# Fetch Batch Details
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during evaluation')
parser.add_argument('--eval_set', type=str, help='Choose from train/valid')

# Other Args
parser.add_argument('--visualize', action='store_true', help='supply this parameter to visualize')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
if FLAGS.visualize:
	BATCH_SIZE = 1
NUM_POINTS = 1024
NUM_EVAL_POINTS = 4096
GT_PCL_SIZE = 4096
NUM_VIEWS = 24
HEIGHT = 128
WIDTH = 128
UP_RATIO = FLAGS.up_ratio

if FLAGS.visualize:
	from utils.show_3d import show3d_balls
	ballradius = 3

def fetch_batch(models, indices, batch_num, batch_size):
	'''
	Input:
		models: list of paths to shapenet models
		indices: list of ind pairs, where 
			ind[0] : model index (range--> [0, len(models)-1])
			ind[1] : view index (range--> [0, NUM_VIEWS-1])
		batch_num: batch_num during epoch
		batch_size: batch size for training or validation
	Returns:
		batch_ip: input RGB image of shape (B, HEIGHT, WIDTH, 3)
		batch_gt: gt point cloud of shape (B, NUM_POINTS, 3)
	Description:
		Batch Loader
	'''
	batch_ip = []
	batch_gt = []
	batch_pcl = []
	pcl_path_gt = []

	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
		model_path = models[ind[0]]
		pcl_ip = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_1024.npy')
		img_path = join(FLAGS.data_dir_imgs, model_path, 'rendering', PNG_FILES[ind[1]])
		#img_path = join(FLAGS.data_dir_imgs, model_path, 'rendering', PNG_FILES[ind[1]])
		pcl_path = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_4096.npy')

		pcl_in = np.load(pcl_ip)
		pcl_gt = np.load(pcl_path)

		ip_image = cv2.imread(img_path)[4:-5, 4:-5, :3]
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)

		batch_pcl.append(pcl_in)
		batch_gt.append(pcl_gt)
		batch_ip.append(ip_image)
		pcl_path_gt.append(pcl_path)

	return np.array(pcl_path_gt), np.array(batch_ip), np.array(batch_gt), np.array(batch_pcl)

def fetch_batch_pix3d(models, batch_num, batch_size):
	''' 
	Inputs:
		models: List of pix3d dicts
		batch_num: batch_num during epoch
		batch_size: batch size for training or validation
	Returns:
		batch_ip: input RGB image of shape (B, HEIGHT, WIDTH, 3)
		batch_gt: gt point cloud of shape (B, NUM_POINTS, 3)
	Description:
		Batch Loader for Pix3D dataset
	'''
	batch_ip = []
	batch_gt = []

	for ind in xrange(batch_num*batch_size,batch_num*batch_size+batch_size):
		_dict = models[ind]
		model_path = '/'.join(_dict['model'].split('/')[:-1])
		model_name = re.search('model(.*).obj', _dict['model'].strip().split('/')[-1]).group(1)
		img_path = join(FLAGS.data_dir_imgs, _dict['img'])
		mask_path = join(FLAGS.data_dir_imgs, _dict['mask'])
		bbox = _dict['bbox'] # [width_from, height_from, width_to, height_to]
		pcl_path_1K = join(FLAGS.data_dir_pcl, model_path,'pcl_%d%s.npy'%(NUM_EVAL_POINTS,model_name))
		ip_image = cv2.imread(img_path)
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
		mask_image = cv2.imread(mask_path)!=0
		ip_image=ip_image*mask_image
		ip_image = ip_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

		current_size = ip_image.shape[:2] # current_size is in (height, width) format
		ratio = float(HEIGHT-PAD)/max(current_size)
		new_size = tuple([int(x*ratio) for x in current_size])
		ip_image = cv2.resize(ip_image, (new_size[1], new_size[0])) # new_size should be in (width, height) format
		delta_w = WIDTH - new_size[1]
		delta_h = HEIGHT - new_size[0]
		top, bottom = delta_h//2, delta_h-(delta_h//2)
		left, right = delta_w//2, delta_w-(delta_w//2)
		color = [0, 0, 0]
		ip_image = cv2.copyMakeBorder(ip_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

		xangle = np.pi/180. * -90
		yangle = np.pi/180. * -90
		pcl_gt = rotate(rotate(np.load(pcl_path_1K), xangle, yangle), xangle)

		batch_gt.append(pcl_gt)
		batch_ip.append(ip_image)

	return np.array(batch_ip), np.array(batch_gt)

def calculate_metrics(models, indices, pcl_gt_1K_scaled, pred_scaled):

	batches = len(indices)/BATCH_SIZE
	if FLAGS.visualize:
		iters = range(batches)
	else:
		iters = tqdm(range(batches))

	epoch_chamfer = 0.
	epoch_forward = 0.
	epoch_backward = 0.
	epoch_emd = 0.	

	ph_gt = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_EVAL_POINTS, 3), name='ph_gt')
	ph_pr = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_EVAL_POINTS, 3), name='ph_pr')

	dists_forward, dists_backward, chamfer_distance = get_chamfer_metrics(ph_gt, ph_pr)
	emd = get_emd_metrics(ph_gt, ph_pr, BATCH_SIZE, NUM_EVAL_POINTS)

	for cnt in iters:
		start = time.time()

		pcl_path, batch_ip, batch_gt_1K, batch_pcl = fetch_batch(models, indices, cnt, BATCH_SIZE)

		_gt_scaled_1K, _pr_scaled = sess.run(
			[pcl_gt_1K_scaled, pred_scaled], 
			feed_dict={pcl_gt_1K:batch_gt_1K, img_inp:batch_ip, pcl_in:batch_pcl}
		)

		_pr_scaled_icp = []
		all_pr_list = []
		for i in xrange(BATCH_SIZE):
			rand_indices = np.random.permutation(GT_PCL_SIZE)[:NUM_EVAL_POINTS]
			T, _, _ = icp(_gt_scaled_1K[i], _pr_scaled[i][rand_indices], tolerance=1e-10, max_iterations=1000)
			_pr_scaled_icp.append(np.matmul(_pr_scaled[i][rand_indices], T[:3,:3]) - T[:3, 3])

		_pr_scaled_icp = np.array(_pr_scaled_icp).astype('float32')

		C,F,B,E = sess.run(
			[chamfer_distance, dists_forward, dists_backward, emd], 
			feed_dict={ph_gt:_gt_scaled_1K, ph_pr:_pr_scaled_icp}
		)

		epoch_chamfer += C.mean() / batches
		epoch_forward += F.mean() / batches
		epoch_backward += B.mean() / batches
		epoch_emd += E.mean() / batches

		if FLAGS.visualize:
			for i in xrange(BATCH_SIZE):
				print '-'*50
				print C[i], F[i], B[i], E[i]
				print '-'*50
				print
				print '*' * 20
				print pcl_path[i]
				print
				batch_ip[i] = np.where(batch_ip[i] > 0, batch_ip[i], 255)
				cv2.imshow('', batch_ip[i])
				print 'Displaying Pr scaled icp 1k'
				print '*' * 20
				all_pr_path = (pcl_path[i])[0:44]
				pcl_path_txt = all_pr_path + 'pcl_path_txt.txt'
				print pcl_path_txt
				print '*' * 20

				all_pr = (pcl_path[i])[44:-20]
				print all_pr
				all_pr_list.append(all_pr)
				with open(pcl_path_txt, 'a') as f:
					for data in all_pr_list:
						f.write(data + '\n')
					f.close()

				print '*' * 20

				pcl_path_npy = (pcl_path[i])[:-19]
				print pcl_path_npy
				print
				pcl_path_pr = pcl_path_npy + '/pointcloud_4096_pr.npy'
				np.save(pcl_path_pr, _pr_scaled_icp[i])
				show3d_balls.showpoints(_pr_scaled_icp[i], ballradius=4)
				print 'PR:', _pr_scaled_icp[i].shape


				print 'Displaying Gt scaled icp 1k'
				show3d_balls.showpoints(_gt_scaled_1K[i], ballradius=4)
				print 'GT:', _gt_scaled_1K[i].shape

		
		if cnt%10 == 0:
			print '%d / %d' % (cnt, batches)

	if not FLAGS.visualize:
		log_values(csv_path, epoch_chamfer, epoch_forward, epoch_backward, epoch_emd)

	return 

def farthest_point_sample(point, npoint):
	"""
	Input:
		xyz: pointcloud data, [N, D]
		npoint: number of samples
	Return:
		centroids: sampled pointcloud index, [npoint, D]
	"""
	B, N, D = point.shape
	print N, D
	xyz = point[:, :3]
	centroids = np.zeros((npoint,))
	distance = np.ones((N,)) * 1e10
	farthest = np.random.randint(0, N)
	for i in range(npoint):
		centroids[i] = farthest
		centroid = xyz[farthest, :]
		dist = np.sum((xyz - centroid) ** 2, -1)
		mask = dist < distance
		distance[mask] = dist[mask]
		farthest = np.argmax(distance, -1)
	point = point[centroids.astype(np.int32)]
	return point


if __name__ == '__main__':

	# Create Placeholders
	img_inp = tf.placeholder(tf.float32, shape=(BATCH_SIZE, HEIGHT, WIDTH, 3), name='img_inp')
	pcl_gt_1K = tf.placeholder(tf.float32, shape=(BATCH_SIZE, GT_PCL_SIZE, 3), name='pcl_gt_1K')
	pcl_in = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, 3), name='pcl_in_1K')

	bn_decay = 0.95
	with tf.variable_scope('psgn') as scope:
		z_latent_img = image_encoder_Ca(img_inp, FLAGS)

		ne = pcl_retrieval(pcl_in, is_training=True, scope=scope,
						reuse=None, use_normal=False, use_bn=False, use_ibn=False,
						bn_decay=bn_decay)
		z_latent_img = tf.concat([z_latent_img, ne], 1)
		print 'z_latent_img', z_latent_img.shape

		out_img = decoder_with_fc_only(z_latent_img, layer_sizes=[512, 1024, np.prod([1024, 3])],
			b_norm=FLAGS.bn_decoder,
			b_norm_finish=False,
			verbose=True,
			scope=scope
			)
	reconstr = tf.reshape(out_img, (BATCH_SIZE, 1024, 3))
	is_training = False
	bn_decay = 0.95
	with tf.variable_scope('generator') as sc:
		# z = encoder_with_convs_and_symmetry(in_signal=reconstr, n_filters=[32, 64, 64],
		# 									filter_sizes=[1], strides=[1],
		# 									b_norm=True, verbose=False)
		#
		# features extraction
		#  global feature
		#  (B, 64) --> (B, 1, 64)
		# global_feat = tf.expand_dims(z, axis=1)
		#  (B, 1, 64) --> (B, NP, 64)
		# global_feat = tf.tile(global_feat, [1, NUM_POINTS, 1])
		# print
		# print('global_feat:', global_feat)
		# print
		# local feature
		# up_l2_points: (B, 1024, 259)
		# up_l2_points, _ = get_local_features(reconstr, is_training=True, scope=sc, bradius = 1.0, reuse=None, use_normal=False,
		# 												use_bn=False, use_ibn=False, bn_decay=bn_decay, up_ratio=UP_RATIO)
		# print
		# print('up_l2_points:', up_l2_points)
		# print
		death_feature = feature_extraction_my(reconstr, scope='spatio_feature_extraction2', is_training=True, bn_decay=None)
		print
		print('death_feature:', death_feature)
		print

		net_last = my_concat_features(death_feature, is_training, scope='up_1_layer', reuse=None, bn_decay=None)

		# out: (B, 4096, 3)
		out = tf.squeeze(net_last, axis=-2)  # (B,rN,3)
		print
		print('out:', out)
		print

		out = decoder_with_convs_only(out, n_filters=[128,128,64,3],
										filter_sizes=[1],
										strides=[1],
										b_norm=True,
										b_norm_finish=False,
										verbose=False)
		print '-' * 30
		print 'out:', out.shape
		print '-' * 30
	#outputs = tf.squeeze(coord, [2])
	#outputs = sample(2048, outputs)  #equals to :outputs = gather_point(outputs, farthest_point_sample(2048, outputs)) 
	out = tf.reshape(out, (BATCH_SIZE, GT_PCL_SIZE, 3))

	# out = farthest_point_sample(out, 1024)

	# Perform Scaling
	pcl_gt_1K_scaled, reconstr_img_scaled = scale(pcl_gt_1K, out)

	# Snapshot Folder Location
	if FLAGS.load_best:
		snapshot_folder = join(FLAGS.exp, 'best')
	else:
		snapshot_folder = join(FLAGS.exp, 'snapshots')

	# Metrics path
	metrics_folder = join(FLAGS.exp, 'metrics', FLAGS.eval_set)
	create_folder(metrics_folder)
	csv_path = join(metrics_folder,'%s.csv'%FLAGS.category)
	with open(csv_path, 'w') as f:
		f.write('Chamfer, Fwd, Bwd, Emd\n')

	# GPU configuration
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

		# Load previous checkpoint
		load_previous_checkpoint(snapshot_folder, saver, sess, is_training=False)

		tflearn.is_training(False, session=sess)

		train_models, val_models, train_pair_indices, val_pair_indices = get_shapenet_models(FLAGS)

		if FLAGS.visualize:
			random.shuffle(val_pair_indices)
			random.shuffle(train_pair_indices)

		if FLAGS.eval_set == 'train':
			calculate_metrics(train_models, train_pair_indices, pcl_gt_1K_scaled, reconstr_img_scaled)
		if FLAGS.eval_set == 'valid':
			calculate_metrics(val_models, val_pair_indices, pcl_gt_1K_scaled, reconstr_img_scaled)
