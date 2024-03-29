from importer import *
#from utils.encoder_2_gen import *
from utils.encoder_2 import *
#from utils.sampling.tf_sampling import farthest_point_sample, gather_point
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir_pcl', type=str, required=True, 
	help='Path to shapenet pointclouds')
parser.add_argument('--exp', type=str, required=True, 
	help='Name of Experiment')
parser.add_argument('--gpu', type=str, required=True, 
	help='GPU to use')
parser.add_argument('--category', type=str, required=True, 
	help='Category to train on : \
		["all", "airplane", "bench", "cabinet", "car", "chair", "lamp", \
		"monitor", "rifle", "sofa", "speaker", "table", "telephone", "vessel"]')
# parser.add_argument('--bottleneck', type=int, required=True, default=512, 
# 	help='latent space size')
parser.add_argument('--up_ratio', type=int, default=4, 
	help='up sampling ratio')
parser.add_argument('--batch_size', type=int, default=32, 
	help='Batch Size during training')
parser.add_argument('--lr', type=float, default=0.0005, 
	help='Learning Rate')
parser.add_argument('--max_epoch', type=int, default=500, 
	help='max num of epoch')
parser.add_argument('--bn_decoder', action='store_true', 
	help='Supply this parameter if you want batch norm in the decoder, otherwise ignore')
parser.add_argument('--print_n', type=int, default=100, 
	help='print output to terminal every n iterations')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size  		# Batch size for training
NUM_POINTS = 1024					# Number of predicted points
GT_PCL_SIZE = 4096				# Number of points in GT point cloud
UP_RATIO = FLAGS.up_ratio


# def fetch_batch(models, batch_num, batch_size):
# 	'''
# 	Input:
# 		models: list of paths to shapenet models
# 		batch_num: batch_num during epoch
# 		batch_size:	batch size for training or validation
# 	Returns:
# 		batch_gt: (B,2048,3)
# 	Description:
# 		Batch Loader
# 	'''
# 	batch_gt = []
# 	batch_in = []
# 	for ind in range(batch_num*batch_size, batch_num*batch_size+batch_size):
# 		model_path = models[ind]
# 		pcl_path = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_1024.npy') # Path to 2K ground truth point cloud
# 		pcl_gt = np.load(pcl_path)
# 		batch_gt.append(pcl_gt)
# 		pcl_path_in = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_1024.npy')
# 		plc_in = np.load(pcl_path_in)
# 		batch_in.append(pcl_in)
# 	batch_gt = np.array(batch_gt)
# 	batch_in = np.array(batch_in)
# 	return batch_gt, batch_in
#  ValueError: setting an array element with a sequence.


def fetch_batch_dense(models, batch_num, batch_size):
	batch_ip = []
	batch_gt = []
	for ind in range(batch_num*batch_size, batch_num*batch_size+batch_size):
		model_path = models[ind]
		pcl_gt = np.load(join(FLAGS.data_dir_pcl, model_path, 'pointcloud_4096.npy'))
		pcl_ip = np.load(join(FLAGS.data_dir_pcl, model_path,'pointcloud_1024.npy'))
		batch_gt.append(pcl_gt)
		batch_ip.append(pcl_ip)
	batch_gt = np.array(batch_gt)
	batch_ip = np.array(batch_ip)
	return batch_ip, batch_gt

def get_epoch_loss(val_models):

	'''
	Input:
		val_models:	list of absolute path to models in validation set
	Returns:
		val_chamfer: chamfer distance calculated on scaled prediction and gt
		val_forward: forward distance calculated on scaled prediction and gt
		val_backward: backward distance calculated on scaled prediction and gt
	Description:
		Calculate val epoch metrics
	'''
	
	tflearn.is_training(False, session=sess)

	batches = len(val_models)/BATCH_SIZE
	val_stats = {}
	val_stats = reset_stats(ph_summary, val_stats)

	for b in xrange(batches):
		batch_ip, batch_gt = fetch_batch_dense(val_models, b, BATCH_SIZE)
		runlist = [loss, chamfer_distance_scaled, dists_forward_scaled, dists_backward_scaled]
		L, C, F, B = sess.run(runlist, feed_dict={pcl_gt:batch_gt, pcl_in:batch_ip})
		_summary_losses = [L, C, F, B]

		val_stats = update_stats(ph_summary, _summary_losses, val_stats, batches)

	summ = sess.run(merged_summ, feed_dict=val_stats)
	return val_stats[ph_dists_chamfer], val_stats[ph_dists_forward], val_stats[ph_dists_backward], summ


if __name__ == '__main__':

	# Create a folder for experiments and copy the training file
	create_folder(FLAGS.exp)
	train_filename = basename(__file__)
	os.system('cp %s %s'%(train_filename, FLAGS.exp))
	with open(join(FLAGS.exp, 'settings.txt'), 'w') as f:
		f.write(str(FLAGS)+'\n')

	# Create Placeholders
	pcl_in = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, 3))
	pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, GT_PCL_SIZE, 3))
	pointclouds_radius = tf.placeholder(tf.float32, shape=(BATCH_SIZE))
	#data_radius = np.ones(shape=(len(pcl_in)))
	#data_radius = np.ones(shape=BATCH_SIZE) TypeError: Expected float for argument 'radius' not <tf.Tensor 'generator/mul:0' shape=(128,) dtype=float32>.

	# Generate Prediction
	#bneck_size = FLAGS.bottleneck
	#is_training = True  pred must not be a Python bool
	is_training = tf.cast(True, tf.bool)
	bn_decay = 0.95

	with tf.variable_scope('generator') as scope:
		global_feature = encoder_with_convs_and_symmetry(in_signal=pcl_in, n_filters=[32,64,64], 
								filter_sizes=[1],
								strides=[1],
								b_norm=True,
								verbose=False
								)	# (bs,64)
		#	global feature
		global_feature = tf.tile(tf.expand_dims(global_feature, axis=1), [1, NUM_POINTS, 1]) # (bs,64) --> (bs,1,64) --> (bs,num_input,64)
		#global_feature = tf.expand_dims(global_feature, axis=2) # (bs,num_input,64) --> (bs,num_input,1,64)
		print '==' * 30
		print 'global_feature:', global_feature.get_shape
		print '==' * 30
		features = feature_extraction(pcl_in, scope='feature_extraction', is_training=True, bn_decay=None)
		print '+' * 30
		print 'feature:', features.get_shape
		print '+' * 30
		up_l2_points, l1_points = get_local_features(pcl_in, is_training=True, scope=scope,reuse=None, use_normal=False, use_bn=False, use_ibn=False,bn_decay=bn_decay,up_ratio=UP_RATIO)
		print '***' * 20
		print 'up_l2_points:', up_l2_points
		print 'l1_points:', l1_points
		print '***' * 20
		net = concat_features(features, up_l2_points, l1_points, global_feature, is_training, scope, reuse=None, bn_decay=None,)
		print '-' * 30
		print 'net old:', net.get_shape
		print '-' * 30
		net = tf.squeeze(net, axis=2)
		print '-' * 30
		print 'net new:', net.get_shape
		print '-' * 30

		# features B,1024,240 ; up_l2_points B,1024,64 ; l1_points B,1024,64 ; global_feature B,1024,64

		outputs = decoder_with_convs_only(net, n_filters=[128,128,64,3], 
										filter_sizes=[1], 
										strides=[1],
										b_norm=True, 
										b_norm_finish=False, 
										verbose=False)
		print '-' * 30
		print 'outputs:', outputs
		print '-' * 30
	#out = tf.squeeze(coord, [2])
	#outputs = sample(2048, outputs)  #equals to :outputs = gather_point(outputs, farthest_point_sample(2048, outputs)) 
	out = tf.reshape(outputs, (BATCH_SIZE, GT_PCL_SIZE, 3))

	print '-' * 30
	print 'out:', out
	print '-' * 30

	# Scale output and gt for val losses
	pcl_gt_scaled, out_scaled = scale(pcl_gt, out)
	
	# Calculate Chamfer Metrics
	dists_forward, dists_backward, chamfer_distance = [tf.reduce_mean(metric) for metric in get_chamfer_metrics(pcl_gt, out)]

	# Calculate Chamfer Metrics on scaled prediction and GT
	dists_forward_scaled, dists_backward_scaled, chamfer_distance_scaled = [tf.reduce_mean(metric) for metric in get_chamfer_metrics(pcl_gt_scaled, out_scaled)]

	# Calculate EMD
	# emd = tf.reduce_mean(get_emd_metrics(pcl_gt, out, BATCH_SIZE, GT_PCL_SIZE))
	# Calculate EMD scaled
	# emd_scaled = tf.reduce_mean(get_emd_metrics(pcl_gt_scaled, out_scaled, BATCH_SIZE, GT_PCL_SIZE))

	# Define Loss to optimize on
	loss = (dists_forward_scaled + dists_backward_scaled/2.0)*10000
	#
	# loss = (dists_forward_scaled + dists_backward_scaled / 2.0) * 10000 + emd_scaled

	# Get Training Models
	train_models, val_models, _, _ = get_shapenet_models(FLAGS)
	batches = len(train_models) / BATCH_SIZE

	# Training Setings
	lr = FLAGS.lr
	optim = tf.train.AdamOptimizer(FLAGS.lr, beta1=0.9).minimize(loss)

	start_epoch = 0
	max_epoch = FLAGS.max_epoch

	# Define Log Directories
	snapshot_folder = join(FLAGS.exp, 'snapshots')
	best_folder = join(FLAGS.exp, 'best')
	logs_folder = join(FLAGS.exp, 'logs')	

	# Define Savers
	saver = tf.train.Saver(max_to_keep=2)

	# Define Summary Placeholders
	ph_loss = tf.placeholder(tf.float32, name='loss')
	ph_dists_chamfer = tf.placeholder(tf.float32, name='dists_chamfer')
	ph_dists_forward = tf.placeholder(tf.float32, name='dists_forward')
	ph_dists_backward = tf.placeholder(tf.float32, name='dists_backward')
	# ph_dists_emd = tf.placeholder(tf.float32, name='dists_emd')


	# ph_summary = [ph_loss, ph_dists_chamfer, ph_dists_forward, ph_dists_backward, ph_dists_emd]
	ph_summary = [ph_loss, ph_dists_chamfer, ph_dists_forward, ph_dists_backward]
	merged_summ = get_summary(ph_summary)

	# Create log directories
	create_folders([snapshot_folder, logs_folder, join(snapshot_folder, 'best'), best_folder])

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:

		train_writer = tf.summary.FileWriter(logs_folder+'/train', sess.graph_def)
		val_writer = tf.summary.FileWriter(logs_folder+'/val', sess.graph_def)

		sess.run(tf.global_variables_initializer())

		# Load Previous checkpoint
		start_epoch = load_previous_checkpoint(snapshot_folder, saver, sess)

		ind = 0
		best_val_loss = 10000000
		since = time.time()

		print '*'*30,'\n','Training Started !!!\n', '*'*30

		PRINT_N = FLAGS.print_n

		for i in xrange(start_epoch, max_epoch): 
			random.shuffle(train_models)
			stats = {}
			stats = reset_stats(ph_summary, stats)
			iter_start = time.time()

			tflearn.is_training(True, session=sess)

			for b in xrange(batches):
				global_step = i*batches + b + 1
				batch_ip, batch_gt = fetch_batch_dense(train_models, b, BATCH_SIZE)
				runlist = [loss, chamfer_distance, dists_forward, dists_backward, optim]
				L, C, F, B, _ = sess.run(runlist, feed_dict={pcl_gt:batch_gt, pcl_in:batch_ip})
				_summary_losses = [L, C, F, B]

				stats = update_stats(ph_summary, _summary_losses, stats, PRINT_N)

				if global_step % PRINT_N == 0:
					summ = sess.run(merged_summ, feed_dict=stats)
					train_writer.add_summary(summ, global_step)
					till_now = time.time() - iter_start
					print 'Loss = {} Iter = {}  Minibatch = {} Time:{:.0f}m {:.0f}s'.format(
						stats[ph_loss], global_step, b, till_now//60, till_now%60
					)
					stats = reset_stats(ph_summary, stats)
					iter_start = time.time()

			print 'Saving Model ....................'
			saver.save(sess, join(snapshot_folder, 'model'), global_step=i)
			print '..................... Model Saved'

			val_epoch_chamfer, val_epoch_forward, val_epoch_backward, val_summ = get_epoch_loss(val_models)
			val_writer.add_summary(val_summ, global_step)

			time_elapsed = time.time() - since

			print '-'*65 + ' EPOCH ' + str(i) + ' ' + '-'*65
			print 'Val Chamfer: {:.8f}  Forward: {:.8f}  Backward: {:.8f}  Time:{:.0f}m {:.0f}s'.format(
				val_epoch_chamfer, val_epoch_forward, val_epoch_backward, time_elapsed//60, time_elapsed%60
			)
			print '-'*140
			print

			if (val_epoch_chamfer < best_val_loss):
				print 'Saving Best at Epoch %d ...............'%(i)
				saver.save(sess, join(snapshot_folder, 'best', 'best'))
				os.system('cp %s %s'%(join(snapshot_folder, 'best/*'), best_folder))
				best_val_loss = val_epoch_chamfer
				print '.............................Saved Best'