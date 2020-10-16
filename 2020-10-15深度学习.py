'''
sigmoid激活函数：除了输出层是一个二分类问题基本不会用它。
tanh激活函数：tanh是非常优秀的，几乎适合所有场合。
ReLu激活函数：最常用的默认函数，，如果不确定用哪个激活函数，就使用ReLu或者Leaky ReLu。

混淆矩阵（也称误差矩阵，Confusion Matrix）:
    真实值是positive，模型认为是positive的数量（True Positive=TP）
    真实值是positive，模型认为是negative的数量（False Negative=FN）：这就是统计学上的第二类错误（Type II Error）
    真实值是negative，模型认为是positive的数量（False Positive=FP）：这就是统计学上的第一类错误（Type I Error）
    真实值是negative，模型认为是negative的数量（True Negative=TN）
    将这四个指标一起呈现在表格中，就能得到如下这样一个矩阵，我们称它为混淆矩阵


注：sotfamx将输出值计算成某一概率，常用于分类问题
回归常用指标：
计算accuracy：tf.metrics.accuracy。 有个博文不错，里面例子不错：https://blog.csdn.net/lyb3b3b/article/details/83047148
tf.metrics.accuracy返回两个值，accuracy为到上一个batch为止的准确度，update_op为更新本批次后的准确度。
accuracy, update_op = tf.metrics.accuracy(labels=x, predictions=y)
计算precision和accuracy差不多
计算sensitivity其实就是计算recall，使用tf.metrics.recall
这里有坑的是specificity，计算这个要基于sensitivity（反过来也可以），使用的是tf.metrics.specificity_at_sensitivity，这个函数相比上面的，多了个参数sensitivity，参考这里：
https://www.w3cschool.cn/tensorflow_python/tensorflow_python-h3v12zap.html
这个参数sensitivity不能用tf.placeholder实现占位，要用大于0小于1的数占位：


https://blog.csdn.net/livan1234/article/details/80875044
一、tensorflow的常用函数：
import tensorflow as tf

import numpy as np

1.1、数据的呈现（Variable（）：定义变量）：

    x=np.array([[1,1,1],[1,-8,1],[1,1,1]])

    w=tf.Variable(initial_value=x)

    w=tf.Variable(tf.zeros([3,3]))

    init=tf.global_variables_initializer()

    withtf.Session() as sess:

           sess.run(init)

           print(sess.run(w))

1.2、数据的加减运算（add():加；multiply():乘）：

    a=tf.placeholder(tf.int16)

    b=tf.placeholder(tf.int16)

    add=tf.add(a,b)

    mul=tf.multiply(a,b)

    withtf.Session() as sess:

           print("a+b=", sess.run(add,feed_dict={a:2, b:3}))

           print("a*b=", sess.run(mul,feed_dict={a:2, b:3}))

1.3、矩阵相乘（matmul）运算：

    a=tf.Variable(tf.ones([3,3]))

    b=tf.Variable(tf.ones([3,3]))

    product=tf.matmul(tf.multiply(5.0,a),tf.multiply(4.0,b))

    init=tf.initialize_all_variables()

    withtf.Session() as sess:

           sess.run(init)

           print(sess.run(product))

1.4、argmax的练习：获取最大值的下标向量

a=tf.get_variable(name='a',shape=[3,4],dtype=tf.float32,initializer=tf.random_uniform_initializer(minval=-1,maxval=1))

    # 最大值所在的下标向量

    b=tf.argmax(input=a,axis=0)

    c=tf.argmax(input=a,dimension=1)

    sess=tf.InteractiveSession()

    sess.run(tf.initialize_all_variables())

    print(sess.run(a))

    print(sess.run(b))

    print(sess.run(c))

1.5、创建全一/全零矩阵：

  tf.ones(shape,type=tf.float32,name=None)

  tf.ones([2, 3], int32) ==> [[1, 1, 1], [1, 1, 1]]

  tf.zeros(shape,type=tf.float32,name=None)

  tf.zeros([2, 3], int32) ==> [[0, 0, 0],[0, 0, 0]]

1.7、tf.ones_like(tensor,dype=None,name=None)

  新建一个与给定的tensor类型大小一致的tensor，其所有元素为1。

    # 'tensor' is [[1, 2, 3], [4, 5, 6]]

    tf.ones_like(tensor) ==> [[1, 1, 1], [1, 1, 1]]

1.8、tf.zeros_like(tensor,dype=None,name=None)

    新建一个与给定的tensor类型大小一致的tensor，其所有元素为0。

    # 'tensor' is [[1, 2, 3], [4, 5, 6]]

    tf.ones_like(tensor) ==> [[0, 0, 0],[0, 0, 0]]

1.9、tf.fill(dim,value,name=None)

     创建一个形状大小为dim的tensor，其初始值为value

    # Output tensor has shape [2, 3].

    fill([2, 3], 9) ==> [[9, 9, 9]

                               [9, 9, 9]]

1.10、tf.constant(value,dtype=None,shape=None,name='Const')

    创建一个常量tensor，先给出value，可以设定其shape

    # Constant 1-D Tensor populated with value list.

    tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 67]

     # Constant 2-D tensor populated with scalarvalue -1.

    tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.] [-1.-1. -1.]

1.11、tf.linspace(start,stop,num,name=None)

     返回一个tensor，该tensor中的数值在start到stop区间之间取等差数列（包含start和stop），如果num>1则差值为(stop-start)/(num-1)，以保证最后一个元素的值为stop。

     其中，start和stop必须为tf.float32或tf.float64。num的类型为int。

    tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.011.0 12.0]

1.12、tf.range(start,limit=None,delta=1,name='range')

     返回一个tensor等差数列，该tensor中的数值在start到limit之间，不包括limit，delta是等差数列的差值。

     start，limit和delta都是int32类型。

    # 'start' is 3

    # 'limit' is 18

    # 'delta' is 3

    tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

    # 'limit' is 5 start is 0

    tf.range(start, limit) ==> [0, 1, 2, 3, 4]

1.13、tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)

    返回一个tensor其中的元素的值服从正态分布。

    seed: A Python integer. Used to create a random seed for thedistribution.See set_random_seed forbehavior。

1.14、tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32,seed=None, name=None)

     返回一个tensor其中的元素服从截断正态分布（？概念不懂，留疑）

1.15、tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)

     返回一个形状为shape的tensor，其中的元素服从minval和maxval之间的均匀分布。

1.16、tf.random_shuffle(value,seed=None,name=None)

    对value（是一个tensor）的第一维进行随机化。

      [[1,2],               [[2,3],

       [2,3],        ==>  [1,2],

       [3,4]]                [3,4]]

1.17、tf.set_random_seed(seed)

    设置产生随机数的种子。

二、常规神经网络（NN）：
import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist=input_data.read_data_sets("Mnist_data/", one_hot=True)

# val_data=mnist.validation.images

# val_label=mnist.validation.labels

#print("______________________________")

# print(mnist.train.images.shape)

# print(mnist.train.labels.shape)

# print(mnist.validation.images.shape)

# print(mnist.validation.labels.shape)

# print(mnist.test.images.shape)

# print(mnist.test.labels.shape)

# print(val_data)

# print(val_label)

# print("==============================")

x = tf.placeholder(tf.float32, [None,784])

y_actual = tf.placeholder(tf.float32, shape=[None,10])

W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

y_predict = tf.nn.softmax(tf.matmul(x, W)+b)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indices=1))

train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(y_predict,1),tf.argmax(y_actual, 1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

init = tf.initialize_all_variables()

with tf.Session() assess:

   sess.run(init)

    fori in range(1000):

       batch_xs, batch_ys = mnist.train.next_batch(100)

       sess.run(train_step, feed_dict={x:batch_xs, y_actual:batch_ys})

       if(i%100==0):

           print("accuracy:" ,sess.run(accuracy, feed_dict={x:mnist.test.images,y_actual:mnist.test.labels}))

三、线性网络模型：
import tensorflow as tf

import numpy as np

# 用numpy随机生成100个数：

x_data=np.float32(np.random.rand(2,100))

y_data=np.dot([0.100,0.200], x_data)+0.300

# 构造一个线性模型：

b=tf.Variable(tf.zeros([1]))

W=tf.Variable(tf.random_uniform([1,2],-1.0, 1.0))

y=tf.matmul(W, x_data)+b

# 最小化方差

loss=tf.reduce_mean(tf.square(y-y_data))

optimizer=tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)

# 初始化变量

init=tf.initialize_all_variables()

# 启动图

sess=tf.Session()

sess.run(init)

# 拟合平面

for step inrange(0, 201):

   sess.run(train)

    ifstep % 20 == 0:

       print(step,sess.run(W), sess.run(b))

四、CNN卷积神经网络：
# -*- coding: utf-8 -*-

"""

Created on ThuSep  8 15:29:48 2016

@author: root

"""

import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data asinput_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None,784])

y_actual = tf.placeholder(tf.float32, shape=[None,10])

# 定义实际x与y的值。

# placeholder中shape是参数的形状，默认为none，即一维数据，[2,3]表示为两行三列；[none，3]表示3列，行不定。

def weight_variable(shape):

   initial = tf.truncated_normal(shape, stddev=0.1)

    returntf.Variable(initial)

# 截尾正态分布，保留[mean-2*stddev, mean+2*stddev]范围内的随机数。用于初始化所有的权值，用做卷积核。

def bias_variable(shape):

   initial = tf.constant(0.1,shape=shape)

    returntf.Variable(initial)

# 创建常量0.1；用于初始化所有的偏置项，即b，用作偏置。

def conv2d(x,W):

    returntf.nn.conv2d(x, W, strides=[1,1, 1,1], padding='SAME')

# 定义一个函数，用于构建卷积层；

# x为input；w为卷积核；strides是卷积时图像每一维的步长；padding为不同的卷积方式；

def max_pool(x):

    returntf.nn.max_pool(x, ksize=[1,2, 2,1], strides=[1,2, 2,1], padding='SAME')

# 定义一个函数，用于构建池化层，池化层是为了获取特征比较明显的值，一般会取最大值max，有时也会取平均值mean。

# ksize=[1,2,2,1]：shape为[batch，height，width， channels]设为1个池化，池化矩阵的大小为2*2,有1个通道。

# strides是表示步长[1,2,2,1]:水平步长为2，垂直步长为2，strides[0]与strides[3]皆为1。

x_image = tf.reshape(x, [-1,28,28,1])

# 在reshape方法中-1维度表示为自动计算此维度，将x按照28*28进行图片转换，转换成一个大包下一个小包中28行28列的四维数组；

W_conv1 = weight_variable([5,5, 1,32])

# 构建一定形状的截尾正态分布，用做第一个卷积核；

b_conv1 = bias_variable([32])

# 构建一维的偏置量。

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+ b_conv1)

# 将卷积后的结果进行relu函数运算，通过激活函数进行激活。

h_pool1 = max_pool(h_conv1)

# 将激活函数之后的结果进行池化，降低矩阵的维度。

W_conv2 = weight_variable([5,5, 32,64])

# 构建第二个卷积核；

b_conv2 = bias_variable([64])

# 第二个卷积核的偏置；

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+ b_conv2)

# 第二次进行激活函数运算；

h_pool2 = max_pool(h_conv2)

# 第二次进行池化运算，输出一个2*2的矩阵，步长是2*2；

W_fc1 = weight_variable([7* 7 * 64,1024])

# 构建新的卷积核，用来进行全连接层运算，通过这个卷积核，将最后一个池化层的输出数据转化为一维的向量1*1024。

b_fc1 = bias_variable([1024])

# 构建1*1024的偏置；

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])

# 对 h_pool2第二个池化层结果进行变形。

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

# 将矩阵相乘，并进行relu函数的激活。

keep_prob = tf.placeholder("float")

# 定义一个占位符。

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 是防止过拟合的，使输入tensor中某些元素变为0，其他没变为零的元素变为原来的1/keep_prob大小，

# 形成防止过拟合之后的矩阵。

W_fc2 = weight_variable([1024,10])

b_fc2 = bias_variable([10])

y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# 用softmax进行激励函数运算，得到预期结果；

# 在每次进行加和运算之后，需要用到激活函数进行转换，激活函数是用来做非线性变换的，因为sum出的线性函数自身在分类中存在有限性。

cross_entropy =-tf.reduce_sum(y_actual*tf.log(y_predict))

# 求交叉熵，用来检测运算结果的熵值大小。

train_step =tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

# 通过训练获取到最小交叉熵的数据，训练权重参数。

correct_prediction =tf.equal(tf.argmax(y_predict,1),tf.argmax(y_actual,1))

accuracy =tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 计算模型的精确度。

sess=tf.InteractiveSession()

sess.run(tf.initialize_all_variables())

for i inrange(20000):

   batch = mnist.train.next_batch(50)

    ifi%100 == 0:

       train_acc = accuracy.eval(feed_dict={x:batch[0],y_actual: batch[1], keep_prob: 1.0})

       # 用括号中的参数，带入accuracy中，进行精确度计算。

       print('step',i,'training accuracy',train_acc)

       train_step.run(feed_dict={x: batch[0],y_actual: batch[1], keep_prob: 0.5})

       # 训练参数，形成最优模型。

test_acc=accuracy.eval(feed_dict={x:mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})

print("test accuracy",test_acc)

Ø  解析：

1）卷积层运算：

# -*- coding: utf-8 -*-

importtensorflow as tf

# 构建一个两维的数组，命名为k。

k=tf.constant([[1,0,1],[2,1,0],[0,0,1]],dtype=tf.float32, name='k')

# 构建一个两维的数组，命名为i。

i=tf.constant([[4,3,1,0],[2,1,0,1],[1,2,4,1],[3,1,0,2]],dtype=tf.float32, name='i')

# 定义一个卷积核，将上面的k形状转化为[3,3,1,1]：长3宽3,1个通道，1个核。

kernel=tf.reshape(k,[3,3,1,1],name='kernel')

# 定义一个原始图像，将上面的i形状转化为[1,4,4,1]：1张图片，长4宽4，1个通道。

image=tf.reshape(i, [1,4,4,1],name='image')

# 用kernel对image做卷积，[1,1,1,1]:每个方向上的滑动步长，此时为四维，故四个方向上的滑动步长全部为1，

sss=tf.nn.conv2d(image, kernel, [1,1,1,1],"VALID")

# 从数组的形状中删除单维条目，即把shape为1的维度去掉，一个降维的过程，得到一个二维的。

res=tf.squeeze(sss)

with tf.Session() assess:

    print(sess.run(k))

    print(sess.run(sss))

    print(sess.run(res))

五、LSTM & GRU

基本LSTM

tensorflow提供了LSTM实现的一个basic版本，不包含lstm的一些高级扩展，同时也提供了一个标准接口，其中包含了lstm的扩展。分别为：tf.nn.rnn_cell.BasicLSTMCell(), tf.nn.rnn_cell.LSTMCell()

LSTM的结构

ensorflow中的BasicLSTMCell()是完全按照这个结构进行设计的

    #tf.nn.rnn_cell.BasicLSTMCell(num_units,forget_bias, input_size, state_is_tupe=Flase, activation=tanh)

cell =tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=1.0, input_size=None,state_is_tupe=Flase, activation=tanh)

    #num_units:图一中ht的维数，如果num_units=10,那么ht就是10维行向量

    #forget_bias：还不清楚这个是干嘛的

    #input_size:[batch_size,max_time, size]。假设要输入一句话，这句话的长度是不固定的，max_time就代表最长的那句话是多长，size表示你打算用多长的向量代表一个word，即embedding_size（embedding_size和size的值不一定要一样）

    #state_is_tuple:true的话，返回的状态是一个tuple:(c=array([[]]), h=array([[]]):其中c代表Ct的最后时间的输出，h代表Ht最后时间的输出，h是等于最后一个时间的output的

    #图三向上指的ht称为output

    #此函数返回一个lstm_cell，即图一中的一个A

如果你想要设计一个多层的LSTM网络，你就会用到tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=False),这里多层的意思上向上堆叠，而不是按时间展开

lstm_cell = tf.nn.rnn_cell.MultiRNNCells(cells,state_is_tuple=False)

    #cells:一个cell列表，将列表中的cell一个个堆叠起来，如果使用cells=[cell]*4的话，就是四曾，每层cell输入输出结构相同

    #如果state_is_tuple:则返回的是 n-tuple，其中n=len(cells): tuple:(c=[batch_size, num_units],h=[batch_size,num_units])

这是，网络已经搭好了，tensorflow提供了一个非常方便的方法来生成初始化网络的state

initial_state =lstm_cell.zero_state(batch_size, dtype=)

    #返回[batch_size, 2*len(cells)],或者[batch_size, s]

    #这个函数只是用来生成初始化值的

现在进行时间展开，有两种方法：
法一：
使用现成的接口：

tf.nn.dynamic_rnn(cell, inputs,sequence_length=None, initial_state=None,dtype=None,time_major=False)

    #此函数会通过，inputs中的max_time将网络按时间展开

    #cell:将上面的lstm_cell传入就可以

    #inputs:[batch_size,max_time, size]如果time_major=Flase. [max_time,batch_size, size]如果time_major=True

    #sequence_length:是一个list，如果你要输入三句话，且三句话的长度分别是5,10,25,那么sequence_length=[5,10,25]

    #返回：（outputs, states）:output，[batch_size, max_time, num_units]如果time_major=False。 [max_time,batch_size,num_units]如果time_major=True。states:[batch_size, 2*len(cells)]或[batch_size,s]

    #outputs输出的是最上面一层的输出，states保存的是最后一个时间输出的states

法二

outputs = []

states = initial_states

with tf.variable_scope("RNN"):

    fortime_step in range(max_time):

       if time_step>0:tf.get_variable_scope().reuse_variables()#LSTM同一曾参数共享，

       (cell_out, state) = lstm_cell(inputs[:,time_step,:], state)

       outputs.append(cell_out)

已经得到输出了，就可以计算loss了,根据你自己的训练目的确定loss函数

tenforflow提供了tf.nn.rnn_cell.GRUCell()构建一个GRU单元

cell = tenforflow提供了tf.nn.rnn_cell.GRUCell(num_units, input_size=None, activation=tanh)
六、常用方法补充：

tf.unstack()
七、tf.nn.softmax_cross_entropy_with_logits的用法：

在计算loss的时候，最常见的一句话就是tf.nn.softmax_cross_entropy_with_logits，那么它到底是怎么做的呢？

首先明确一点，loss是代价值，也就是我们要最小化的值.

tf.nn.softmax_cross_entropy_with_logits(logits,labels, name=None)

除去name参数用以指定该操作的name，与方法有关的一共两个参数：

第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes

第二个参数labels：实际的标签，大小同上

具体的执行流程大概分为两步：

第一步是先对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，对于单样本而言，输出就是一个num_classes大小的向量（[Y1，Y2,Y3...]其中Y1，Y2，Y3...分别代表了是属于该类的概率）

至于为什么是用的这个公式？这里不介绍了，涉及到比较多的理论证明

第二步是softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵，公式如下：

其中,指代实际的标签中第i个的值（用mnist数据举例，如果是3，那么标签是[0，0，0，1，0，0，0，0，0，0]，除了第4个值为1，其他全为0）

就是softmax的输出向量[Y1，Y2,Y3...]中，第i个元素的值

显而易见，预测越准确，结果的值越小（别忘了前面还有负号），最后求一个平均，得到我们想要的loss

注意！！！这个函数的返回值并不是一个数，而是一个向量，如果要求交叉熵，我们要再做一步tf.reduce_sum操作,就是对向量里面所有元素求和，最后才得到   ，如果求loss，则要做一步tf.reduce_mean操作，对向量求均值！

最后上代码:

import tensorflow as tf

#our NN's output

logits=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])

#step1:do softmax

y=tf.nn.softmax(logits)

#true label

y_=tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])

#step2:do cross_entropy

cross_entropy =-tf.reduce_sum(y_*tf.log(y))

#do cross_entropy just one step

cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits,y_))#dont forget tf.reduce_sum()!!

with tf.Session() as sess:

   softmax=sess.run(y)

   c_e = sess.run(cross_entropy)

   c_e2 = sess.run(cross_entropy2)

   print("step1:softmax result=")

   print(softmax)

   print("step2:cross_entropy result=")

   print(c_e)

   print("Function(softmax_cross_entropy_with_logits)result=")

   print(c_e2)

输出结果是：
step1:softmax result=

[[ 0.09003057  0.24472848 0.66524094]

 [0.09003057  0.24472848  0.66524094]

 [0.09003057  0.24472848  0.66524094]]

step2:cross_entropy result=

1.22282

Function(softmax_cross_entropy_with_logits)result=

1.2228
最后大家可以试试e^1/(e^1+e^2+e^3)是不是0.09003057，发现确实一样！！这也证明了我们的输出是符合公式逻辑的

八、RNN应用：

Ø  RNN案例（一）

# -*- coding: utf-8 -*-

import tensorflow as tf

importtensorflow.examples.tutorials.mnist.input_data as input_data

mnist =input_data.read_data_sets("MNIST_data/", one_hot=True)

lr = 0.001

training_iters = 100000

batch_size = 128

n_inputs = 28

n_steps = 28

n_hidden_units = 128

n_classes = 10

# 生成两个占位符；

x = tf.placeholder(tf.float32, [None,n_steps, n_inputs])

y = tf.placeholder(tf.float32, [None,n_classes])

weights = {

    # 随机生成一个符合正态图形的矩阵，作为in和out的初始值。

   'in':tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),

   'out':tf.Variable(tf.random_normal(n_hidden_units, n_classes)),

    }
biases = {

   'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),

   'out':tf.Variable(tf.constant(0.1, shape=[n_classes, ])),

    }

def RNN(X, weights, biases):

    # 第一步：输入的x为三维数据，因此需要进行相应的维度变换；转换成2维，然后与w、b进行交易，运算完成后，再将x转换成三维；

   X=tf.reshape(X, [-1, n_inputs])

   X_in = tf.matmul(X, weights['in'])+biases['in']

   X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # 第二步：即构建cell的初始值，并进行建模运算；

    #n_hidden_units:是ht的维数，表示128维行向量；state_is_tuple表示tuple形式，返回一个lstm的单元，即一个ht。

   lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units,forget_bias=1.0, state_is_tuple=True)

    # 将LSTM的状态初始化全为0数组，batch_size给出一个batch大小。

   init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # 运算一个神经单元的输出值与状态，动态构建RNN模型，在这个模型中实现ht与x的结合。

   outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in,initial_state=init_state, time_major=False

    # 第三步：将输出值进行格式转换，然后运算输出，即可。

    # 矩阵的转置，[0,1,2]为正常顺序[高，长，列]，想要更换哪个就更换哪个的顺序即可,并实现矩阵解析。

   outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))

   results = tf.matmul(outputs[-1], weights['out']) + biases['out']

   return results

# 创建一个模型，然后进行测试。

pred = RNN(x, weights, biases)

# softmax_cross_entropy_with_logits：将神经网络最后一层的输出值pred与实际标签y作比较，然后计算全局平均值，即为损失。

cost =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

# 用梯度下降优化，下降速率为0.001。

train_op =tf.train.AdamOptimizer(lr).minimize(cost)

# 计算准确度。

correct_pred = tf.equal(tf.argmax(pred, 1),tf.argmax(y, 1))

accuracy =tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

   sess.run(init)

   step = 0

   while step*batch_size < training_iters:

       batch_xs, batch_ys = mnist.train.next_batch(batch_size)

       batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])

       sess.run([train_op], feed_dict={

           x:batch_xs,

           y:batch_ys,

           })

        if step % 20 ==0:

           print(sess.run(accuracy, feed_dict={

                x:batch_xs,

                y:batch_ys,

                }))

           step += 1

Ø  RNN案例（二）

# num_epochs = 100

# total_series_length = 50000

# truncated_backprop_length = 15

# state_size = 4

# num_classes = 2

# echo_step = 3

# batch_size = 5

# num_batches =total_series_length//batch_size//truncated_backprop_length

# def generateData():

#     x= np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))

#     y= np.roll(x, echo_step)

#    y[0:echo_step] = 0
#     x= x.reshape((batch_size, -1))  # Thefirst index changing slowest, subseries as rows

#     y= y.reshape((batch_size, -1))

#    return (x, y)

# batchX_placeholder =tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])

# batchY_placeholder =tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

# init_state = tf.placeholder(tf.float32,[batch_size, state_size])

# W =tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)

# b = tf.Variable(np.zeros((1,state_size)),dtype=tf.float32)
# W2 = tf.Variable(np.random.rand(state_size,num_classes),dtype=tf.float32)

# b2 = tf.Variable(np.zeros((1,num_classes)),dtype=tf.float32)

# # Unpack columns

# inputs_series =tf.unstack(batchX_placeholder, axis=1)

# labels_series =tf.unstack(batchY_placeholder, axis=1)
# # Forward pass

# current_state = init_state

# states_series = []

# for current_input in inputs_series:

#    current_input = tf.reshape(current_input, [batch_size, 1])

#    input_and_state_concatenated = tf.concat(1, [current_input,current_state])  # Increasing number ofcolumns
#    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) +b)  # Broadcasted addition

#    states_series.append(next_state)

#    current_state = next_state

# logits_series = [tf.matmul(state, W2) + b2for state in states_series] #Broadcasted addition

# predictions_series = [tf.nn.softmax(logits)for logits in logits_series]
# losses =[tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits,labels in zip(logits_series,labels_series)]

# total_loss = tf.reduce_mean(losses)
# train_step =tf.train.AdagradOptimizer(0.3).minimize(total_loss)

# def plot(loss_list, predictions_series,batchX, batchY):

#    plt.subplot(2, 3, 1)

#    plt.cla()

#    plt.plot(loss_list)
#    for batch_series_idx in range(5):

#        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx,:]

#        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for outin one_hot_output_series])
#        plt.subplot(2, 3, batch_series_idx + 2)

#        plt.cla()

#        plt.axis([0, truncated_backprop_length, 0, 2])

#        left_offset = range(truncated_backprop_length)

#        plt.bar(left_offset, batchX[batch_series_idx, :], width=1,color="blue")

#        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1,color="red")

#        plt.bar(left_offset, single_output_series * 0.3, width=1,color="green")
#    plt.draw()

#    plt.pause(0.0001)
# with tf.Session() as sess:

#    sess.run(tf.initialize_all_variables())

#    plt.ion()

#    plt.figure()

#    plt.show()

#    loss_list = []

# for epoch_idx in range(num_epochs):

#    x,y = generateData()

#    _current_state = np.zeros((batch_size, state_size))

#    print("New data, epoch", epoch_idx)

# for batch_idx in range(num_batches):

#    start_idx = batch_idx * truncated_backprop_length

#    end_idx = start_idx + truncated_backprop_length

#    batchX = x[:,start_idx:end_idx]

#    batchY = y[:,start_idx:end_idx]
#    _total_loss, _train_step, _current_state, _predictions_series =sess.run(

#    [total_loss, train_step, current_state, predictions_series],

#    feed_dict={

#    batchX_placeholder:batchX,

#    batchY_placeholder:batchY,

#     init_state:_current_state

#    })

#    loss_list.append(_total_loss)

#    if batch_idx%100 == 0:

#        print("Step",batch_idx, "Loss", _total_loss)

#        plot(loss_list, _predictions_series, batchX, batchY)

#    plt.ioff()

#    plt.show()
Ø  LSTM_RNN案例（三）：

# -*-coding: utf-8 -*-

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from asn1crypto._ffi import null

BATCH_START=0     #建立batch data的时候的index

TIME_STEPS=20     #backpropagation throughtime的time_steps

BATCH_SIZE=50     #

INPUT_SIZE=1      #sin数据输入size

OUTPUT_SIZE=1     #cos数据输入size

CELL_SIZE=10      #RNN的hiden unit size

LR=0.006         #学习率

# 定义一个生成数据的get_batch的function：

def get_batch():

    global BATCH_START, TIME_STEPS

    xs=np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE)

        .reshape((BATCH_SIZE,TIME_STEPS))/(10*np.pi)

    seq=np.sin(xs)

    res=np.cos(xs)

    BATCH_START+=TIME_STEPS

    # np.newaxis:在功能上等价于none；

    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]

class LSTMRNN(object):

    def __init__(self, n_steps, input_size, output_size, cell_size,batch_size):

        self.n_steps=n_steps

        self.input_size=input_size

        self.output_size=output_size

        self.cell_size=cell_size

        self.batch_size=batch_size

        # 构建命名空间，在inputs命名空间下的xs和ys与其他空间下的xs和ys是不冲突的，一般与variable一起用。

        with tf.name_scope('inputs'):

            self.xs=tf.placeholder(tf.float32,[None, n_steps, input_size], name='xs')

            self.ys=tf.placeholder(tf.float32,[None, n_steps, output_size], name='ys')

        # variable_scope与get_variable()一起用，实现变量共享，指向同一个内存空间。

      with tf.variable_scope('in_hidden'):

            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op=tf.train.AdamOptimizer(LR).minimize(self.cost)
    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
        Ws_in=self._weight_variable([self.input_size, self.cell_size])
        bs_in=self._bias_variable([self.cell_size,])
        with tf.name_scope('Wx_plus_b'):
            l_in_y=tf.matmul(l_in_x,Ws_in)+bs_in
        self.l_in_y=tf.reshape(l_in_y,[-1, self.n_steps, self.cell_size],name='2_3D')

    def add_cell(self):
        lstm_cell=tf.contrib.rnn.BasicLSTMCell(self.cell_size,forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state=lstm_cell.zero_state(self.batch_size,dtype=tf.float32)
        self.cell_outputs,self.cell_final_state=tf.nn.dynamic_rnn(lstm_cell, self.l_in_y,initial_state=self.cell_init_state, time_major=False)
    def add_output_layer(self):
        l_out_x=tf.reshape(self.cell_outputs,[-1, self.cell_size], name='2_2D')
        Ws_out=self._weight_variable([self.cell_size,self.output_size])
        bs_out=self._bias_variable([self.output_size,])
        with tf.name_scope('Wx_plus_b'):
            self.pred=tf.matmul(l_out_x,Ws_out)+bs_out
    # 求交叉熵
    def compute_cost(self):
       losses=tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size*self.n_steps],dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
            )
        with tf.name_scope('average_cost'):
            self.cost=tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost',
            )
            tf.summary.scalar('cost',self.cost)
    def ms_error(self,labels,logits):
        #求方差
        return tf.square(tf.subtract(labels,logits))
    def _weight_variable(self, shape, name='weights'):
       initializer=tf.random_normal_initializer(mean=0, stddev=1.,)
        return tf.get_variable(shape=shape,initializer=initializer, name=name)
    def _bias_variable(self, shape, name='biases'):
        initializer=tf.constant_initializer(0, 1)
        return tf.get_variable(name=name, shape=shape,initializer=initializer)
if __name__=='__main__':
    model=LSTMRNN(TIME_STEPS, INPUT_SIZE,OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    state=null
    for i in range(200):
        seq, res, xs=get_batch()
        if i == 0:
            feed_dict={
                model.xs:seq,
                model.ys:res,
                }
        else:
            feed_dict={
                model.xs:seq,
                model.ys:res,
                model.cell_init_state:state
                }

        _, cost, state, pred=sess.run(
            [model.train_op, model.cost,model.cell_final_state, model.pred],
            feed_dict=feed_dict)
        if i%20==0:
            print('cost:',round(cost, 4))
九、使用flags定义命令行参数
import tensorflow as tf
第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.app.flags.DEFINE_string('str_name', 'def_v_1',"descrip1")
tf.app.flags.DEFINE_integer('int_name', 10,"descript2")
tf.app.flags.DEFINE_boolean('bool_name', False, "descript3")
FLAGS = tf.app.flags.FLAGS

#必须带参数，否则：'TypeError: main() takes no arguments (1given)';   main的参数名随意定义，无要求
defmain(_):
    # 在这个函数中添加脚本所需要处理的内容。
    print(FLAGS.str_name)
    print(FLAGS.int_name)
    print(FLAGS.bool_name)
if __name__ == '__main__':
   tf.app.run()  #执行main函数
注：
FLAGS命令是指编写一个脚本文件，在执行这个脚本时添加相应的参数；
如（上面文件叫tt.py）：
python tt.py --str_name test_str--int_name 99 --bool_name True
十、tensor变换：
#对于2-D
# Tensor变换主要是对矩阵进行相应的运算工作，包涵的方法主要有：reduce_……（a, axis）系列；如果不加axis的话都是针对整个矩阵进行运算。
tf.reduce_sum(a, 1）#对axis1
tf.reduce_mean(a,0) #每列均值
第二个参数是axis，如果为0的话，res[i]=∑ja[j,i]res[i]=∑ja[j,i]即（res[i]=∑a[:,i]res[i]=∑a[:,i]）， 如果是1的话，res[i]=∑ja[i,j]res[i]=∑ja[i,j]
NOTE:返回的都是行向量,（axis等于几，就是对那维操作,i.e.:沿着那维操作, 其它维度保留）
#关于concat，可以用来进行降维 3D->2D , 2D->1D
tf.concat(concat_dim, data)
#arr = np.zeros([2,3,4,5,6])
In [6]: arr2.shape

Out[6]: (2, 3, 4, 5)
In [7]: np.concatenate(arr2, 0).shape
Out[7]: (6, 4, 5)   :(2*3, 4, 5)
In [9]: np.concatenate(arr2, 1).shape
Out[9]: (3, 8, 5)   :(3, 2*4, 5)
#tf.concat()
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
# 将t1, t2进行concat，axis为0，等价于将shape=[2,2, 3]的Tensor concat成
#shape=[4, 3]的tensor。在新生成的Tensor中tensor[:2,:]代表之前的t1
#tensor[2:,:]是之前的t2
tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

# 将t1, t2进行concat，axis为1，等价于将shape=[2,2, 3]的Tensor concat成
#shape=[2, 6]的tensor。在新生成的Tensor中tensor[:,:3]代表之前的t1
#tensor[:,3:]是之前的t2
tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
concat是将list中的向量给连接起来，axis表示将那维的数据连接起来，而其他维的结构保持不变
#squeeze 降维维度为1的降掉
tf.squeeze(arr, [])
降维，将维度为1的降掉
arr = tf.Variable(tf.truncated_normal([3,4,1,6,1], stddev=0.1))
arr2 = tf.squeeze(arr, [2,4])
arr3 = tf.squeeze(arr) #降掉所以是1的维
# split(dimension, num_split, input)：dimension的意思就是输入张量的哪一个维度，如果是0就表示对第0维度进行切割。num_split就
是切割的数量，如果是2就表示输入张量被切成2份，每一份是一个列表。
tf.split(split_dim, num_split, value, name='split')
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(1, 3, value)
tf.shape(split0) ==> [5, 10]
#embedding: embedding_lookup是按照向量获取矩阵中的值，[0,2,3,1]是取第0,2,3,1个向量。
mat = np.array([1,2,3,4,5,6,7,8,9]).reshape((3,-1))
ids = [[1,2], [0,1]]
res = tf.nn.embedding_lookup(mat, ids)
res.eval()
array([[[4, 5, 6],
        [7, 8, 9]],
       [[1, 2, 3],
        [4, 5, 6]]])
# expand_dims：扩展维度，如果想用广播特性的话，经常会用到这个函数
# 't' is a tensor of shape [2]
#一次扩展一维
shape(tf.expand_dims(t, 0)) ==> [1, 2]
shape(tf.expand_dims(t, 1)) ==> [2, 1]
shape(tf.expand_dims(t, -1)) ==> [2, 1]
# 't2' is a tensor of shape [2, 3, 5]
shape(tf.expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(tf.expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(tf.expand_dims(t2, 3)) ==> [2, 3, 5, 1]
tf.slice()
tf.slice(input_, begin, size, name=None)
这个函数的作用是从输入数据input中提取出一块切片
o   切片的尺寸是size，切片的开始位置是begin。
o   切片的尺寸size表示输出tensor的数据维度，其中size[i]表示在第i维度上面的元素个数。
o   开始位置begin表示切片相对于输入数据input_的每一个偏移量
import tensorflow as tf
import numpy as np
sess = tf.Session()
input=tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
data = tf.slice(input, [1, 0, 0], [1, 1, 3])
print(sess.run(data))
"""[1,0,0]表示第一维偏移了1 则是从[[[3, 3, 3],[4, 4, 4]],[[5, 5, 5], [6, 6, 6]]]中选取数据然后选取第一维的第一个，第二维的第一个数据，第三维的三个数据"""
# [[[3 3 3]]]
#array([[ 6,  7],
#       [11, 12]])
理解tf.slice()最好是从返回值上去理解，现在假设input的shape是[a1, a2, a3], begin的值是[b1, b2, b3],size的值是[s1, s2, s3],那么tf.slice()返回的值就是 input[b1:b1+s1,b2:b2+s2, b3:b3+s3]。
如果 si=−1si=−1 ，那么 返回值就是 input[b1:b1+s1,...,bi: ,...]
注意：input[1:2] 取不到input[2]
tf.stack()
tf.stack(values, axis=0, name=’stack’)
tf.stack（）这是一个矩阵拼接的函数，tf.unstack（）则是一个矩阵分解的函数
将 a list of R 维的Tensor堆成 R+1维的Tensor。
Given a list of length N of tensors of shape (A, B, C);
if axis == 0 then the output tensor will have the shape (N, A, B, C)
这时 res[i,:,:,:] 就是原 list中的第 i 个 tensor
if axis == 1 thenthe output tensor will have the shape (A, N, B, C).
这时 res[:,i,:,:] 就是原list中的第 i 个 tensor
Etc.
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
stack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
stack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
tf.gather()：按照指定的下标集合从axis=0中抽取子集
tf.gather(params, indices, validate_indices=None,name=None)
·        tf.slice(input_,begin, size, name=None)：按照指定的下标范围抽取连续区域的子集
·        tf.gather(params,indices, validate_indices=None, name=None)：按照指定的下标集合从axis=0中抽取子集，适合抽取不连续区域的子集
# Scalar indices, 会降维

output[:, ..., :] = params[indices, :, ... :]

# Vector indices

output[i, :, ..., :] = params[indices[i], :, ... :]

# Higher rank indices，会升维

output[i, ..., j, :, ... :] = params[indices[i, ..., j],:, ..., :]

tf.pad

tf.pad(tensor, paddings, mode="CONSTANT", name=None)

·        tensor: 任意shape的tensor，维度 Dn

·        paddings: [Dn, 2] 的 Tensor, Padding后tensor的某维上的长度变为padding[D,0]+tensor.dim_size(D)+padding[D,1]

·        mode: CONSTANT表示填0, REFLECT表示反射填充,SYMMETRIC表示对称填充。

·        函数原型：
tf.pad
pad(tensor, paddings, mode=’CONSTANT’, name=None)
（输入数据，填充的模式，填充的内容，名称）
·        这里紧紧解释paddings的含义：
它是一个Nx2的列表，
在输入维度n上，则paddings[n,0] 表示该维度内容前面加0的个数, 如对矩阵来说，就是行的最上面或列最左边加几排0
paddings[n, 1] 表示该维度内容后加0的个数。
·        举个例子：
输入 tensor 形如
[[ 1, 2],
[1, 2]]
而paddings = [[1, 1], [1, 1]] （[[上，下],[左，右]]）
Tensor=[[1,2],[1,2]]
Paddings=[[1,1],[1,1]]
Init=tf.global_variables_initializer()
Withtf.Session() as sess:
Sess.run(init)
Print(sess.run(tf.pad(tensor,paddings, mode=’CONSTANT’)))
则结果为:
[[0, 0, 0, 0],
[0, 1, 2, 0],
[0, 1, 2, 0],
[0, 0, 0, 0]
十一、损失函数：
损失函数的运算规则：







'''