'''
需要一个或者两个隐藏层来处理（对于MINST数据集，一个拥有数百个神经元的隐藏层就可以达到97%的精度，而用同样数量神经元构建的两层隐藏层就可以获得
超过98%的精度，而且训练时间基本相同）。对于更复杂的问题，你可以逐渐增减隐藏层的层次，直到在训练集上产生过度拟合。非常复杂的问题，比如大图片
的分类，或者语音识别，通常需要数十层的隐藏层（甚至数百层，非全连接的层，我们将在第13章讨论），当然它们需要超大的训练数据集。不过，很少会从头
构建这样的网络：更常见的是重用别人训练好的用来处理类似任务的网络。这样训练就会快很多，而且需要的数据量也会少很多。
通常来说，通过增加每层的神经元数量比增加层数会产生更多的消耗

一个更简单的做法是使用（比实际所需）更多的层次和神经元，然后提前结束训练来避免过度拟合（以及其他的正则化技术，特别是
dropout，我们将在第11章讨论）。这被称为“弹力裤”方法。[1]无须花费时间找刚好适合你的裤子，随便挑弹力裤，它会缩小到合适的尺寸。

激活函数：
    通常来说ELU函数>leaky ReLU函数（和它的变种）>ReLU函数>tanh函数>逻辑函数。如果你更关心运行时的性能，那你可以选择leaky ReLU函数，
    而不是ELU函数。如果你不想改变别的超参数，就只使用建议α的默认值（leaky ReLU函数是0.01，ELU函数是1）。如果你有多余的时间和计算
    能力，你可以使用交叉验证去评估别的激活函数，特别是如果你的网络过度拟合，你可以使用RReLU函数，又或者是针对大的训练集使用PReLU函
    数。
    线性整流函数（Rectified Linear Unit, ReLU），又称修正线性单元，是一种人工神经网络中常用的激活函数（activation function），通常
    指代以斜坡函数及其变种为代表的非线性函数

批量归一化:梯度下降方法很简单，但是它有个毛病，就是需要我们人为的去选择参数，比如学习率、参数初始化、权重衰减系数、Drop out比例等
    不需要过拟合调参，可以选择很大的学习率，可以将数据顺序打乱，不需要局部归一化

tf.variance_scaling_initializer() tensorflow学习：参数初始化
tf.layers.dense()全连接层，相当于添加一个层
tf.clip_by_value(array,a,b)将一个数组或者张量限制在a,b之间


'''

import tensorflow as tf
import numpy as np


# Leaky ReLU漏斗修正线性单元
def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

# elu修正线性单元
def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
print(y_valid, y_train)

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)# 如果是漏斗修正线性单元或者elu的话，改为tf.nn.leaky_relu
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# n_epochs = 100
# batch_size = 50

n_epochs = 20
batch_size = 50
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

    save_path = saver.save(sess, "./my_model_final.ckpt")