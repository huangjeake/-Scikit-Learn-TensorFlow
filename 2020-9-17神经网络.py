'''
感知器：
    感知器是人工神经网络中的一种典型结构， 它的主要的特点是结构简单，对所能解决的问题 存在着收敛算法，
    并能从数学上严格证明，从而对神经网络研究起了重要的推动作用
    和逻辑回归分类器相反，感知器不输出某个类概率。它只能根据一个固定的阈值来做预测 单个感知器无法解决异或分类问题
    异或分类:将数字转化为二进制的，不同的为0，相同的部分为1

反向传播算法：
    反向传播算法先做一次预测（正向过程），度量误差，然后反向的遍历每个层次来度量每个连接的误差贡献度（反向过程），最
    后再微调每个连接的权重来降低误差（梯度下降）。

简单神经网络模型步骤：
    dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], n_classes=10,
                                     feature_columns=feature_cols)# 创建评估模型

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)# 设置需要输入的训练集数据
    dnn_clf.train(input_fn=input_fn)# 用数据训练模型
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_test}, y=y_test, shuffle=False)# 设置需要输入的测试集数据
    eval_results = dnn_clf.evaluate(input_fn=test_input_fn)# 评估模型

# 自定义neuron_layer函数,神经网络层函数：
import tensorflow as tf
n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])# 获取第二维
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)# 设置标准差，使算法更快收敛
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")# 设置偏差b
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")# 设置连接层
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)# labels归一化之后输出的数量 logits神经网络输出的数据
    loss = tf.reduce_mean(xentropy, name="loss")
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
n_epochs = 40



tf.keras.datasets.mnist.load_data() 从亚马逊服务器导入数据
tf.feature_column.numeric_column 数值列
tf.estimator.inputs.numpy_input_fn(
    x,# x ：numpy數組對像或numpy數組對象的dict。 如果是數組，則該數組將被視為單個特徵。
    y=None, # y ：numpy數組對像或numpy數組對象的dict。 None如果沒有。
    batch_size=128,# batch_size ：整數，要返回的批次大小。
    num_epochs=1,# num_epochs ：整數，迭代數據的紀元數。 如果None將永遠運行。
    shuffle=None,# shuffle ：Boolean，如果為True，則對隊列進行洗牌。 在預測時避免隨機播放。
    queue_capacity=1000,# queue_capacity ：整數，要累積的隊列大小。
    num_threads=1 # num_threads ：整數，用於讀取和排隊的線程數。 為了具有預測和可重複的讀取和排隊順序，例如在預測和評估模式中， num_threads應為1。
)
tf.truncated_normal(shape, mean, stddev)# 参数1生成树组形状；参数二均值；参数三标准差 截断的正太分布，如果值大于均值的两倍标准差则重新生成
tf.nn.in_top_k主要是用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量，tf.nn.in_top_k(prediction, target, K):prediction就是表示
你预测的结果，大小就是预测样本的数量乘以输出的维度，类型是tf.float32等。target就是实际样本类别的标签，大小就是样本数量的个数。K表示每个样本
的预测结果的前K个最大的数里面是否含有target中的值。一般都是取1。

tf.cast数据类型转换

'''
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
# a = tf.estimator.inputs.numpy_input_fn()