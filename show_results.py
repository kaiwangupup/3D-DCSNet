
from importer import *
from tqdm import tqdm
import cv2

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

# Machine Details
parser.add_argument('--gpu', type=str, required=True, help='[Required] GPU to use')


# parser.add_argument('--data_dir', type=str, required=True, help='Path to shapenet rendered images')

# Experiment Details
parser.add_argument('--mode', type=str, required=True, help='[Required] Latent Matching setup. Choose from [lm, plm]')
parser.add_argument('--exp', type=str, required=True, help='[Required] Path of experiment for loading pre-trained model')
parser.add_argument('--load_best', action='store_true', help='load best val model')

# AE Details
parser.add_argument('--bottleneck', type=int, required=False, default=512, help='latent space size')
# parser.add_argument('--bn_encoder', action='store_true', help='Supply this parameter if you want bn_encoder, otherwise ignore')
parser.add_argument('--bn_decoder', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')
parser.add_argument('--bn_decoder_final', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')

parser.add_argument('--eval_set', type=str, help='Choose from train/valid')
parser.add_argument('--image', type=str, help='Choose from train/valid')
#parser.add_argument('--txt', type=str, help='Choose from train/valid')

# Other Args
parser.add_argument('--visualize', action='store_true', help='supply this parameter to visualize')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

if FLAGS.visualize:
	BATCH_SIZE = 1

NUM_POINTS = 1024
NUM_VIEWS = 24
HEIGHT = 128
WIDTH = 128
PAD = 35
INPUT = FLAGS.image

if FLAGS.visualize:
	from utils.show_3d import show3d_balls
	ballradius = 3
#----------------visualize----------------
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)
 
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
 
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        #axis('off')
        #title('feature_map_{}'.format(i))
 
    plt.savefig('feature_map.png')
    plt.show()
 

    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")

def save_conv_img(conv_img):
    feature_maps = np.squeeze(conv_img,axis=0)
    img_num = feature_maps.shape[2]
    all_feature_maps = []
    for i in range(0,img_num):
        single_feature_map = feature_maps[:,:,i]
        all_feature_maps.append(single_feature_map)
        cv2.imwrite('/home/ubuntu/fig' + 'feature_{}'.format(i) + '.png',single_feature_map)
        #plt.imshow(single_feature_map)
        #plt.savefig('feature_{}'.format(i))
        
    sum_feature_map = sum(feature_map for feature_map in all_feature_maps)
    cv2.imshow('feature_img',sum_feature_map)
    cv2.imwrite('/home/ubuntu/fig/final.png',sum_feature_map)



if __name__ == '__main__':

	ip_image = cv2.imread(INPUT)[4:-5, 4:-5, :3]
	ip_image = cv2.resize(ip_image,(128,128),interpolation=cv2.INTER_CUBIC)
	ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
	batch_ip = []
	batch_ip.append(ip_image)
	batch_ip = np.array(batch_ip)
	img_inp = tf.placeholder(tf.float32, shape=(1,HEIGHT, WIDTH, 3), name='img_inp')
	
	# Generate Prediction
	with tf.variable_scope('psgn') as scope:
		z_latent_img, fig1, fig2 = image_encoder_se_pure_vis(img_inp, FLAGS)

		out_img = decoder_with_fc_only(z_latent_img, layer_sizes=[512,1024,np.prod([NUM_POINTS, 3])],
			b_norm=FLAGS.bn_decoder,
			b_norm_finish=False,
			verbose=True,
			scope=scope
			)
	reconstr = tf.reshape(out_img, (BATCH_SIZE, NUM_POINTS, 3))
	bneck_size = FLAGS.bottleneck
	with tf.variable_scope('pointnet_ae') as scope:
		z = encoder_with_convs_and_symmetry(in_signal=reconstr, n_filters=[64,128,128,256,bneck_size], 
			filter_sizes=[1],
			strides=[1],
			b_norm=True,
			verbose=True,
			scope=scope
			)
		out = decoder_with_fc_only(z, layer_sizes=[512,1024,np.prod([NUM_POINTS, 3])],
			b_norm=FLAGS.bn_decoder,
			b_norm_finish=False,
			verbose=True,
			scope=scope
			)
		reconstr_img = tf.reshape(out, (BATCH_SIZE, NUM_POINTS, 3))

	# Perform Scaling
	_, reconstr_img_scaled = scale(reconstr_img,reconstr_img)
	#reconstr_img_scaled = reconstr_img

	#visualize
	#visualize_feature_map(fig1)
	
	if FLAGS.load_best:
		snapshot_folder = join(FLAGS.exp, 'best')
	else:
		snapshot_folder = join(FLAGS.exp, 'snapshots')

	 # GPU configuration
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		load_previous_checkpoint(snapshot_folder, saver, sess, is_training=False)
		tflearn.is_training(False, session=sess)
		if FLAGS.eval_set == 'valid':
			conv_img ,_pred_scaled = sess.run([fig1, reconstr_img_scaled],feed_dict={img_inp:batch_ip})
			print conv_img.shape
			save_conv_img(conv_img)
			if FLAGS.visualize:
				for i in xrange(BATCH_SIZE):
					cv2.imshow('', batch_ip[0])
					print 'Displaying Pr scaled icp 1k'
					#print _pred_scaled
					show3d_balls.showpoints(_pred_scaled[i], ballradius=3)
			

		else:
			print 'Invalid dataset. Choose from [shapenet, pix3d]'
			sys.exit(1)
