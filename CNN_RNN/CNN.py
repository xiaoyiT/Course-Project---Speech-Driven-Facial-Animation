import tensorflow as tf
import numpy as np

def cnn_model_fn(features, labels, mode):

    input_layer = tf.reshape(features["x"],[-1,128,23,1])

    conv1 = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[3, 1],padding="same",activation=tf.nn.relu,strides=[2,1])
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 1], strides=[2,1])
    conv2 = tf.layers.conv2d(inputs=pool1,filters=32,kernel_size=[3, 1],padding="same",activation=tf.nn.relu,strides=[2,1])
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 1], strides=[2,1])
    conv3 = tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=[2, 1],padding="same",activation=tf.nn.relu,strides=[2,1])
    conv4 = tf.layers.conv2d(inputs=conv3,filters=64,kernel_size=[3, 1],padding="same",activation=tf.nn.relu,strides=[2,1])
    conv5 = tf.layers.conv2d(inputs=conv4,filters=128,kernel_size=[3, 1],padding="same",activation=tf.nn.relu,strides=[2,1])
    

    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[1, 2], strides=[1,2])
    conv6 = tf.layers.conv2d(inputs=pool3,filters=128,kernel_size=[1, 3],padding="same",activation=tf.nn.relu,strides=[1,2])
    conv7 = tf.layers.conv2d(inputs=conv6,filters=136,kernel_size=[1, 3],padding="same",activation=tf.nn.relu,strides=[1,2])
    conv8 = tf.layers.conv2d(inputs=conv7,filters=136,kernel_size=[1, 4],padding="same",activation=tf.nn.relu,strides=[1,4])


    conv8_flat = tf.reshape(conv8, [-1,136])
    dense1 = tf.layers.dense(inputs=conv8_flat, units=136 , activation=tf.nn.tanh)
    rnn = tf.nn.rnn_cell.GRUCell(dense1)
    dense2 = tf.layers.dense(inputs=dense1, units=136 , activation=tf.nn.tanh)
    output = tf.layers.dense(inputs=dense2, units=136 , activation=tf.nn.sigmoid)

    predictions = {"classes": output}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels,output)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"error":tf.metrics.mean_squared_error(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)