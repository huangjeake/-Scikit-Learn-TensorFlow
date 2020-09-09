import tensorflow as tf
# 导入tensorflow

def weight_variable(shape):
    # 在正太分布中截取形状为shape的随机数
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# 这里可以初步理解为k
# 我们设立一个y=kx+b的方程，我们导入数据x，y，有很多个这样二元一次方程被设立
# 根据这个二元一次方程组可以求出k和b的值，然后带入新的数据x，则可以求出结果y

def bias_variable(shape):
    # 创建常量
    initial = tf.constant(0.1, shape=shape)
    # 生成变量
    return tf.Variable(initial)
# 同理这里可以初步理解为b

def conv2d(x, W):
    # x 指需要做卷积的输入图像 y 相当于CNN中的卷积核
    # strides 卷积在每一维的步长，一般为一个一维向量，长度为4
    # padding 定义元素边框和元素内容之间的空间，只能是‘SAME’（边缘填充）或者‘VALID’（边缘不填充）。
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
# 转换为2d

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# 设置最大为2x2
# 先构建会话再定义操作
sess = tf.InteractiveSession()
# 设置sess

# placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存
x = tf.placeholder("float", shape=[None, 784])
# 设置X输入数组
x_image = tf.reshape(x, [-1,28,28,1])
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
# 计算激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# 设置激活函数
y_ = tf.placeholder("float", shape=[None, 10])
# 设置结果y
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 设置训练步长
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# 设置正确预测
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 设置结果精确度

sess.run(tf.initialize_all_variables())
# 运行

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0}
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
#不断训练模型，根新参数

feed_dict={x:mnist.test.images, y_: mnist.test.labels,keep_prob:1.0}
print("test accuracy %g" % accuracy.eval(feed_dict=feed_dict))
# 输出结果