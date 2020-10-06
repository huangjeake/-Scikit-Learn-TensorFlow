# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/")
# # reset_graph()
# import sys
# from functools import partial
#
#
# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 150  # codings 编码层 其他隐藏层对称
# n_hidden3 = n_hidden1
# n_outputs = n_inputs
#
# learning_rate = 0.01
# l2_reg = 0.0001
# X = tf.placeholder(tf.float32, shape=[None, n_inputs])
#
# he_init = tf.contrib.layers.variance_scaling_initializer()
# l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
# my_dense_layer = partial(tf.layers.dense,
#                          activation=tf.nn.elu,
#                          kernel_initializer=he_init,
#                          kernel_regilarizer=l2_regularizer)
# hidden1 = my_dense_layer(X, n_hidden1)
# hidden2 = my_dense_layer(hidden1, n_hidden2)
# hidden3 = my_dense_layer(hidden2, n_hidden3)
# outputs = my_dense_layer(hidden3, n_outputs, activation=None)
# reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
# reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# loss = tf.add_n([reconstruction_loss] + reg_loss)
#
# optimizer = tf.train.AdamOptimizer(learning_rate)
# training_op = optimizer.minimize(loss)
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# n_epochs = 5
# batch_size = 150
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         n_batches = mnist.train.num_examples // batch_size
#         for iteration in range(n_batches):
#             print("\r{}%".format(100 * iteration // n_batches), end="")  # not shown in the book
#             sys.stdout.flush()  # not shown
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             sess.run(training_op, feed_dict={X: X_batch})
#         loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})  # not shown
#         print("\r{}".format(epoch), "Train MSE:", loss_train)  # not shown
#         saver.save(sess, "./my_model_all_layers.ckpt")
import tensorflow as tf
import numpy as np
import gym

# reset_graph()

n_inputs = 4
n_hidden = 4
n_outputs = 1

learning_rate = 0.01

initializer = tf.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)
P_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(P_left_and_right), num_samples=1)
y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)
