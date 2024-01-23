import tensorflow as tf
from tflearn.layers.conv import global_avg_pool

#the ratio prefer set to 4
def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        scale = input_x * excitation

        return scale


def Global_Average_Pooling(x):
    return global_avg_pool(x, name="SE_Global_average_pooling")

def Fully_connected(x, units=class_num, layer_name='SE_fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

#----- the split and transform module -----

x = split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i), is_training)
x = transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i), is_training)
x = squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio, layer_name='squeeze_layer_'+layer_num+'_'+str(i))

def split_layer(self, input_x, stride, layer_name, is_training):
    with tf.name_scope(layer_name) :
        layers_split = list()
        for i in range(cardinality) :  #3 or 6 ？ 自己设定的C,在ResNeXt中设置为32
            splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i), training)
            layers_split.append(splits)

        return Concatenation(layers_split)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def transform_layer(x,depth, stride, scope, is_training):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=depth, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=is_training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=[3,3], stride=stride, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=is_training, scope=scope+'_batch2')
            x = Relu(x)
            return x

def transition_layer( x, out_dim, scope, is_training):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=is_training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

def Batch_Normalization(x, scope, is_training):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=is_training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=is_training, reuse=True))