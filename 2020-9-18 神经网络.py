'''
需要一个或者两个隐藏层来处理（对于MINST数据集，一个拥有数百个神经元的隐藏层就可以达到97%的精度，而用同样数量神经元构建的两层隐藏层就可以获得
超过98%的精度，而且训练时间基本相同）。对于更复杂的问题，你可以逐渐增减隐藏层的层次，直到在训练集上产生过度拟合。非常复杂的问题，比如大图片
的分类，或者语音识别，通常需要数十层的隐藏层（甚至数百层，非全连接的层，我们将在第13章讨论），当然它们需要超大的训练数据集。不过，很少会从头
构建这样的网络：更常见的是重用别人训练好的用来处理类似任务的网络。这样训练就会快很多，而且需要的数据量也会少很多。
通常来说，通过增加每层的神经元数量比增加层数会产生更多的消耗

一个更简单的做法是使用（比实际所需）更多的层次和神经元，然后提前结束训练来避免过度拟合（以及其他的正则化技术，特别是
dropout，我们将在第11章讨论）。这被称为“弹力裤”方法。[1]无须花费时间找刚好适合你的裤子，随便挑弹力裤，它会缩小到合适的尺寸。

神经网络提高训练速度的方法：
    在连接权重上应用一个良好的初始化策略，使用一个良好的激活函数，使用批量归一化，以及重用部分预处理网络。另一种明显提
    高训练速度的方法是使用快速优化器

激活函数：
    通常来说ELU函数>leaky ReLU函数（和它的变种）>ReLU函数>tanh函数>逻辑函数。如果你更关心运行时的性能，那你可以选择leaky ReLU函数，
    而不是ELU函数。如果你不想改变别的超参数，就只使用建议α的默认值（leaky ReLU函数是0.01，ELU函数是1）。如果你有多余的时间和计算
    能力，你可以使用交叉验证去评估别的激活函数，特别是如果你的网络过度拟合，你可以使用RReLU函数，又或者是针对大的训练集使用PReLU函
    数。
    线性整流函数（Rectified Linear Unit, ReLU），又称修正线性单元，是一种人工神经网络中常用的激活函数（activation function），通常
    指代以斜坡函数及其变种为代表的非线性函数

批量归一化:梯度下降方法很简单，但是它有个毛病，就是需要我们人为的去选择参数，比如学习率、参数初始化、权重衰减系数、Drop out比例等
    不需要过拟合调参，可以选择很大的学习率，可以将数据顺序打乱，不需要局部归一化
    批量归一化流程：隐藏曾 -- 批量归一化 -- 激活函数
    作用：减少参数的数量

梯度裁剪：减轻梯度爆炸的问题；

重用tensorflow模型：https://blog.csdn.net/wang_jiankun/article/details/81135774
    一：全部重用： 法1：直接拷贝构建模型的代码；法2：恢复模型图 恢复模型的图 获得模型的op 在session中恢复模型数据 保存节点加载模型，
                        用数据训练完模型之后保存新的模型
    二：部分重用：法一：重新构建图，恢复部分模型数据;1、重新构建图 添加新的隐藏层，其他数据保持一致 2、恢复部分模型数据

                    # 创建saver指定要恢复的数据，注意这个saver的名称和保存现在的模型的saver不能一样
                    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope="hidden[123]") # regular expression
                    restore_saver = tf.train.Saver(reuse_vars) # to restore layers 1-3

                    # 保存现在的模型的saver
                    saver = tf.train.Saver()

                    # 恢复模型部分数据
                    with tf.Session() as sess:
                        restore_saver.restore(sess, "./my_model_final.ckpt")
                 法二：恢复图，修改图，恢复模型数据
                    # 恢复图
                    saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")
                    # 获得所需的op，注意名称
                    X = tf.get_default_graph().get_tensor_by_name("X:0")
                    y = tf.get_default_graph().get_tensor_by_name("y:0")
                    hidden3 = tf.get_default_graph().get_tensor_by_name("dnn/hidden4/Relu:0")
                    # 要添加层的参数
                    n_hidden4 = 20
                    n_outputs = 10
                    # 添加层
                    new_hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="new_hidden4")
                    new_logits = tf.layers.dense(new_hidden4, n_outputs, name="new_outputs")

从其他框架复用模型：如果只有模型的权重数据没有图，可以自己构建图，然后加载权重到相应的层。

冻结低层：
    # with tf.name_scope('train'):
    #     optimizer = tf.train.GradientDescentOptimizer(learning_ratea)
    #     train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hidden[34]outputs')
    #     training_op = optimizer.minimize(loss, var_list=train_vars)

缓存冻结层的含义:
    基于上述的冻结层，我们已经将原有的模型的第一层与第二层进行了冻结，那么表示他的权重并不发生变化。如果现在训练新的网络数据仍然从第一层开始，
    这样会增加很多计算的开销，由于前两层的权重已经不会改变了，这时明智的做法是将第二层的输出作为新的网络的输入。
    首先，可以一次性在这两个底层跑完全部的数据集：
    hidden2_outputs=sess.run(hidden2,feed_dict={X:X_train})
    然后，在训练中，批量构建上述隐藏层2的输出(hidden2_outputs).
    import numpy as np
    n_epoch=100
    n_batches=500
    for epoch in range(n_epoch):
        shuffled_idx=rnd.permutation(len(hidden2_outputs))
        hidden2_batches=np.array_split(hidden2_outputs[shuffled_idx],n_batches)
        y_batches=np.array_split(y_train[shuffled_idx],n_batches)
        for hidden2_batch,y_batch in zip(hidden2_batches, y_batches):
            sess.run(training_op,feed_dict={hidden2: hidden2_batch, y: y_batch})

优化器：快速优化器
    Momentum optimization optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    Nesterov Accelerated  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9, use_nesterov=True)
    Gradient AdaGrad  optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    RMSProp Adam  optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum=0.9, decay=0.9, epsilon=1e-10)
    Optimization  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

学习计划：相比于固定的学习率 刚开始较大的学习率 后面较小的学习率表现更好
    因为AdaGrad、RMSProp和Adam优化在训练中自动降低了学习速率，所以不需要额外加入学习计划
    设置学习率衰减
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
    decay_steps, decay_rate)# 学习率衰减函数 每一万步学习率减少0.1
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss, global_step=global_step)

通过正则化防止过拟合：不要忘记将正则化损失加到整体损失中，不然就会被直接忽略掉。
    方式一：
    通过get_tensor_by_name获得权重，通过l1正则化超参数，基础损失加上权重加权的偏移生成新的损失
    方式二：
    通过设置tf.GraphKeys的正则化损失
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name='loss')

通过dropout防止过拟合：
    模型过度拟合，你可以提高dropout速率（即降低keep_prob超参数）。相反，如果模型不拟合训练集，你需要降低dropout速率（即提高keep_prob超参数）
    输入层和每一个隐藏层的输出调用dropout（）函数
    dropout_rate = 0.5  # == 1 - keep_prob
    X_drop = tf.layers.dropout(X, dropout_rate, training=training)
    hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,
                              name="hidden1")
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu,
                              name="hidden2")
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

通过最大范数正则化：
    方式一：通过梯度裁剪获得梯度裁剪权重
    threshold = 1.0
    clipped_weights = tf.clip_by_norm(weights, clip_norm=threshold, axes=1)
    clip_weights = tf.assign(weights, clipped_weights)
    在添加数据的时候取值
    clip_weights.eval()
    方式二：构建max_norm_regularizer（）函数
    def max_norm_regularizer(threshold, axes=1, name="max_norm",
         collection="max_norm"):
         def max_norm(weights):
         clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
         clip_weights = tf.assign(weights, clipped, name=name)
         tf.add_to_collection(collection, clip_weights)
         return None # there is no regularization loss term
         return max_norm



tf.variance_scaling_initializer() tensorflow学习：参数初始化
tf.layers.dense()全连接层，相当于添加一个层
tf.clip_by_value(array,a,b)将一个数组或者张量限制在a,b之间
tf.placeholder_with_default(input, shape, name=None) 一个占位符的操作，当它的输出没有被馈送时，通过input传递

tf.layers.batch_normalization(
    inputs,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer=tf.zeros_initializer(),
    gamma_initializer=tf.ones_initializer(),
    moving_mean_initializer=tf.zeros_initializer(),
    moving_variance_initializer=tf.ones_initializer(),
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    training=False,
    trainable=True,
    name=None,
    reuse=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    virtual_batch_size=None,
    adjustment=None
)
axis的值取决于按照input的哪一个维度进行BN，例如输入为channel_last format，即[batch_size, height, width, channel]，则axis应该设定为4，
如果为channel_first format，则axis应该设定为1.
momentum的值用在训练时，滑动平均的方式计算滑动平均值moving_mean和滑动方差moving_variance。 后面做更详细的说明。
center为True时，添加位移因子beta到该BN层，否则不添加。添加beta是对BN层的变换加入位移操作。注意，beta一般设定为可训练参数，即trainable=True。
scale为True是，添加缩放因子gamma到该BN层，否则不添加。添加gamma是对BN层的变化加入缩放操作。注意，gamma一般设定为可训练参数，即trainable=True。
training表示模型当前的模式，如果为True，则模型在训练模式，否则为推理模式。要非常注意这个模式的设定，这个参数默认值为False。如果在训练时采用了
默认值False，则滑动均值moving_mean和滑动方差moving_variance都不会根据当前batch的数据更新，这就意味着在推理模式下，均值和方差都是其初始值，因为这两个值并没有在训练迭代过程中滑动更新。

tf.nn.elu(features)如果features小于0,计算指数线性：exp(features) - 1,否则为features.

from functools import partial  partial 函数允许给一个或者多个参数设置固定值
tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)# labels归一化之后输出的数量 logits神经网络输出的数据
tf.GraphKeys.UPDATE_OPS，这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作
tf.get_collection 从集合中取出
tf.get_default_graph().get_tensor_by_name("X:0")# 表示X节点第一个输出张量
tf.Graph.get_operation_by_name(name)# 根据名称返回操作节点
tf.train.import_meta_graph用来加载meta文件中的图,以及图上定义的结点参数包括权重偏置项等需要训练的参数,也包括训练过程生成的中间
参数,所有参数都是通过graph调用接口get_tensor_by_name(name="训练时的参数名称")来获取
tf.GraphKeys.GLOBAL_VARIABLES 默认加入所有变量对象，并且在分布式中共享
tf.train.exponential_decay指数函数
tf.clip_by_norm 梯度裁剪

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


# 带有批量归一化和激活函数的神经网络
from functools import partial
batch_norm_momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')
with tf.name_scope('dnn'):
    he_init = tf.variance_scaling_initializer()
    my_batch_norm_layer = partial(
        tf.layers.batch_normalization,
        training=training,
        momentum=batch_norm_momentum
    )
    my_dense_layer=partial(
        tf.layers.dense,
        kernel_initializer=he_init
    )
    hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
    hidden2 = my_dense_layer(bn1, n_hidden2, name='hidden2')
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
    logits_before_bn = my_dense_layer(bn2, n_outputs, name='outputs')
    logits = my_batch_norm_layer(logits_before_bn)
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
with tf.name_scope('eval'):
#     correct = tf.nn.in_top_k(logits, y, 1)
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 20
batch_size = 200

threshold = 1.0

# 梯度裁剪的方法减轻梯度爆炸
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# grads_and_vars = optimizer.compute_gradients(loss)# 计算梯度
# capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)# 裁剪梯度 使之保持在一个区间内
#               for grad, var in grads_and_vars]
# training_op = optimizer.apply_gradients(capped_gvs)# 与optimizer.minimize(loss)一样

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)# 这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run([training_op, extra_update_ops],
                     feed_dict={training: True, X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")
