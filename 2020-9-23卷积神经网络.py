'''

卷积神经网络：全连接层，S型激活函数，卷积层，池化层
    接受野：对视野的局部区域做出反应的范围
    步幅:两个连续的接受野之间的距离叫作步幅
    例子：10 * 10的输入层连接到 4 * 6层 感受野是7 * 5   步幅6 * 4每个方向可以相同 也可以不同


特征图公式：
    ((n+2p−f)/s  +1)∗((n+2p−f)s  +1) 其中n为原始图像大小,p为Padding填充维度,f为卷积核维度,s为步长


tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None)
input:
    指需要做卷积的输入图像（tensor），具有[batch,in_height,in_width,in_channels]这样的4维shape，分别是图片数量、图片高度、图
    片宽度、图片通道数，数据类型为float32或float64。
filter:
    相当于CNN中的卷积核，它是一个tensor，shape是[filter_height,filter_width,in_channels,out_channels]：滤波器高度、宽度、图像
    通道数、滤波器个数，数据类型和input相同。
strides:
    卷积在每一维的步长，一般为一个一维向量，长度为4，一般为[1,stride,stride,1]。
padding:
    定义元素边框和元素内容之间的空间，只能是‘SAME’（边缘填充）或者‘VALID’（边缘不填充）。
return：
    返回值是Tensor

最大池化：
tf.nn.max_pool(
            h,
            ksize=[1, height, width, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
h : 需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch_size, height, width, channels]这样的shape
k_size : 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
strides : 窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
padding： 填充的方法，SAME或VALID，SAME表示添加全0填充，VALID表示不添加

'''
# import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
#
# height = 28
# width = 28
# channels = 1
# n_inputs = height * width
#
# conv1_fmaps = 32
# conv1_ksize = 3
# conv1_stride = 1
# conv1_pad = "SAME"
#
# conv2_fmaps = 64
# conv2_ksize = 3
# conv2_stride = 2
# conv2_pad = "SAME"
#
# pool3_fmaps = conv2_fmaps
#
# n_fc1 = 64
# n_outputs = 10
#
# # reset_graph()
#
# with tf.name_scope("inputs"):
#     X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
#     X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#     y = tf.placeholder(tf.int32, shape=[None], name="y")
#
# conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
#                          strides=conv1_stride, padding=conv1_pad,
#                          activation=tf.nn.relu, name="conv1")
# conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
#                          strides=conv2_stride, padding=conv2_pad,
#                          activation=tf.nn.relu, name="conv2")
#
# with tf.name_scope("pool3"):
#     pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#     pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])
#
# with tf.name_scope("fc1"):
#     fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
#
# with tf.name_scope("output"):
#     logits = tf.layers.dense(fc1, n_outputs, name="output")
#     Y_proba = tf.nn.softmax(logits, name="Y_proba")
#
# with tf.name_scope("train"):
#     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
#     loss = tf.reduce_mean(xentropy)
#     optimizer = tf.train.AdamOptimizer()
#     training_op = optimizer.minimize(loss)
#
# with tf.name_scope("eval"):
#     correct = tf.nn.in_top_k(logits, y, 1)
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# with tf.name_scope("init_and_save"):
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
# X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
# X_valid, X_train = X_train[:5000], X_train[5000:]
# y_valid, y_train = y_train[:5000], y_train[5000:]
#
# def shuffle_batch(X, y, batch_size):
#     rnd_idx = np.random.permutation(len(X))
#     n_batches = len(X) // batch_size
#     for batch_idx in np.array_split(rnd_idx, n_batches):
#         X_batch, y_batch = X[batch_idx], y[batch_idx]
#         yield X_batch, y_batch
#
# n_epochs = 10
# batch_size = 100
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#         print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
#
#         save_path = saver.save(sess, "./my_mnist_model")

