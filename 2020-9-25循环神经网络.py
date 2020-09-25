'''


处理长度可变输入序列：
    basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
    tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,sequence_length=seq_length)# 添加一个参数sequence_length


LSTM单元：解决多个时间迭代训练的难点，训练一个长序列的RNN  相关信息和预测信息的位置较远，RNN无法处理长期依赖的问题
    长短期记忆网络通常被称为“LSTM”，是一种特殊的RNN，能够学习长期依赖性
    如果将LSTM单元视为黑盒，那么除了性能比较好之外，它用起来就和一个基本单元一样。训练将更快收敛，并且能检测数据中的长期依赖。
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)

GRU单元是LSTM的简化版本，它的表现和LSTM差不多：
    gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)



tf.contrib.rnn.BasicRNNCell(num_units)# 创建一个RNN实例，num_units参数指代了有多少个隐藏单元

tf.stack（）这是一个矩阵拼接的函数，tf.unstack（）则是一个矩阵分解的函数

tf.transpose(a, perm = None, name = 'transpose')将a进行转置，并且根据perm参数重新排列输出维度

tf.nn.rnn_cell.OutputProgectionWrapper()将rnn_cell的输出映射成想要的维度

tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)函数是tensorflow封装的用来实现递归神经网络（RNN）的函数
参数：basic_cell：RNN实例，X是输入
返回值：元组（outputs, states）outputs输出 states表示最终的状态

'''

#训练序列分类器
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# reset_graph()

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

X_test = X_test.reshape((-1, n_steps, n_inputs))
n_epochs = 100
batch_size = 150
print(len(X_train), len(y_train))

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
