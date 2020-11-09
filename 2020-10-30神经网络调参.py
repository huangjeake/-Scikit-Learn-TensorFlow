
import tensorflow as tf
# tf.nn.sigmoid
# tf.nn.relu
# tf.glorot_uniform_initializer()
# tf.truncated_normal_initializer()
# tf.keras.initializers.he_normal()
# tf.train.AdamOptimizer
# tf.train.MomentumOptimizer
# tf.train.GradientDescentOptimizer
# saver = tf.train.Saver()
# saver.save()
# saver.restore()

'''
Tensorboard实战：
    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", acc)
    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    # Summarize all gradients
    for grad, var in grads:
        tf.summary.histogram(var.name + '/gradient', grad)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())
    _, c, summary = sess.run([apply_grads, loss, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
    终端输入 tensorboard --logdir=./log
fine_tune实战:
    保存模型：1.saver = tf.train.Saver() saver.save(sess, 模型路径) 2.joblib
    模型加载：1.saver.restore(sess, 模型路径)2.joblib
    冻结神经网络层： 卷积层trainable设置为False，不进行训练，保存原始参数
                        
activation-initializer-optimizer：
    tf.nn.sigmoid# s激活函数
    tf.nn.relu# 修正线性单元
    tf.glorot_uniform_initializer
    tf.truncated_normal_initializer
    tf.keras.initializers.he_normal
    tf.train.AdamOptimizer
    tf.train.MomentumOptimizer
    tf.train.GradientDescentOptimizer
        
图像增强实战：
    distorted_image=tf.random_crop(reshape_image,[height,width,3])#从原图像中切割出子图像  
    distorted_image=tf.image.random_flip_left_right(distorted_image) #随机地左右翻转图像
    distorted_images=tf.image.random_brightness(distorted_image,max_delta=63)#随机调节图像的亮度
    distorted_image=tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)#随机地调整图像对比度
    float_image=tf.image.per_image_whitening(distorted_image)#对图像进行whiten操作，目的是降低输入图像的冗余性，尽量去除输
    入特征间的相关性

神经网络过拟合解决方法：
    1.增加数据量
    2.降低模型复杂度
    3.多个模型加权评估
    4.增加dropout层

批量归一化：
    conv -- bn -- relu
    在卷积层激活函数设置NONE，归一化卷积层输出，在归一化函数中设置trainning = True or False
    在卷积封装函数设置is_trainning的值
    设置id_training 占位符 tf.placeholder(tf.bool,[])
    在sess.run中喂数据的时候，添加is_training:True 或者 is_trainning:False
    

'''

import tensorflow as tf
import os
import numpy as np
import pickle

# 文件存放目录
CIFAR_DIR = "./MK/cifar-10-batches-py"

# tensorboard
# 1. 指定面板图上显示的变量
# 2. 训练过程中将这些变量计算出来,输出到文件中
# 3. 文件解析 ./tensorboard  --logdir = dir.

def load_data( filename ):
    '''read data from data file'''
    with open( filename, 'rb' ) as f:
        data = pickle.load( f, encoding='bytes' ) # python3 需要添加上encoding='bytes'
        return data[b'data'], data[b'labels'] # 并且 在 key 前需要加上 b

class CifarData:
    def __init__( self, filenames, need_shuffle ):
        '''参数1:文件夹 参数2:是否需要随机打乱'''
        all_data = []
        all_labels = []

        for filename in filenames:
            # 将所有的数据,标签分别存放在两个list中
            data, labels = load_data( filename )
            all_data.append( data )
            all_labels.append( labels )

        # 将列表 组成 一个numpy类型的矩阵!!!!
        self._data = np.vstack(all_data)
        # 对数据进行归一化, 尺度固定在 [-1, 1] 之间
        self._data = self._data
        # 将列表,变成一个 numpy 数组
        self._labels = np.hstack( all_labels )
        # 记录当前的样本 数量
        self._num_examples = self._data.shape[0]
        # 保存是否需要随机打乱
        self._need_shuffle = need_shuffle
        # 样本的起始点
        self._indicator = 0
        # 判断是否需要打乱
        if self._need_shuffle:
            self._shffle_data()

    def _shffle_data( self ):
        # np.random.permutation() 从 0 到 参数,随机打乱
        p = np.random.permutation( self._num_examples )
        # 保存 已经打乱 顺序的数据
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch( self, batch_size ):
        '''return batch_size example as a batch'''
        # 开始点 + 数量 = 结束点
        end_indictor = self._indicator + batch_size
        # 如果结束点大于样本数量
        if end_indictor > self._num_examples:
            if self._need_shuffle:
                # 重新打乱
                self._shffle_data()
                # 开始点归零,从头再来
                self._indicator = 0
                # 重新指定 结束点. 和上面的那一句,说白了就是重新开始
                end_indictor = batch_size # 其实就是 0 + batch_size, 把 0 省略了
            else:
                raise Exception( "have no more examples" )
        # 再次查看是否 超出边界了
        if end_indictor > self._num_examples:
            raise Exception( "batch size is larger than all example" )

        # 把 batch 区间 的data和label保存,并最后return
        batch_data = self._data[self._indicator:end_indictor]
        batch_labels = self._labels[self._indicator:end_indictor]
        self._indicator = end_indictor
        return batch_data, batch_labels

# 拿到所有文件名称
train_filename = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
# 拿到标签
test_filename = [os.path.join(CIFAR_DIR, 'test_batch')]

# 拿到训练数据和测试数据
train_data = CifarData( train_filename, True )
test_data = CifarData( test_filename, False )

batch_size = 20
# 设计计算图
# 形状 [None, 3072] 3072 是 样本的维数, None 代表位置的样本数量
x = tf.placeholder( tf.float32, [batch_size, 3072] )
# 形状 [None] y的数量和x的样本数是对应的
y = tf.placeholder( tf.int64, [batch_size] )

is_training = tf.placeholder(tf.bool, [])


x_image = tf.reshape( x, [-1, 3, 32, 32] )
# 将最开始的向量式的图片,转为真实的图片类型
x_image = tf.transpose( x_image, perm= [0, 2, 3, 1] )

# 将x_image拆分,eg: [(1, 32, 32, 3), (1, 32, 32, 3), ... ]
x_image_arr = tf.split(x_image, num_or_size_splits = batch_size, axis = 0)

# 用于存储增强后的图片
result_x_image_arr = []

for x_single_image in x_image_arr:
    # 将x_single_image改变形状,改为图片的格式. eg:[1, 32, 32, 3] -> [32, 32, 3]
    x_single_image = tf.reshape(x_single_image, [32, 32, 3])
    # 上下反转
    data_aug_1 = tf.image.random_flip_left_right(x_single_image)
    # 增加亮度
    data_aug_2 = tf.image.random_brightness(data_aug_1, max_delta = 63)
    # 增加对比度
    data_aug_3 = tf.image.random_contrast(data_aug_2, lower = 0.2, upper = 1.8)

    # 将单张图片重新改成 四维
    x_single_image = tf.reshape(data_aug_3, [1, 32, 32, 3])

    # 将单张图片存入列表
    result_x_image_arr.append(x_single_image)

# 将result_x_image_arr重新合并成数据集的样子
result_x_images = tf.concat(result_x_image_arr, axis = 0)

# 重新做归一化
normal_result_x_images = result_x_images / 127.5 - 1

"""
def conv_wrapper(inputs,
                 name,
                 output_channel = 32,
                 kernel_size = (3, 3),
                 activation = tf.nn.relu,
                 padding = 'same'):
    '''
    tf.layers.conv2d 的包裹函数
    :param inputs:
    :param name:
    :param output_channel:
    :param kernel_size:
    :param activation:
    :param padding:
    :return:
    '''
    return tf.layers.conv2d(inputs,
                            output_channel,
                            kernel_size,
                            padding = padding,
                            activation = activation,
                            name = name)
"""

def conv_wrapper(inputs,
                 name,
                 is_training,
                 output_channel = 32,
                 kernel_size = (3, 3),
                 activation = tf.nn.relu,
                 padding = 'same'):
    '''
    卷积层 包裹函数
    :param inputs:
    :param name:
    :param is_training:
    :param output_channel:
    :param kernel_size:
    :param activation:
    :param padding:
    :return:
    '''
    # without bn: conv -> activation
    # with batch normalization: conv -> bn -> activation

    with tf.name_scope(name):
        conv2d = tf.layers.conv2d(inputs,
                                output_channel,
                                kernel_size,
                                padding = padding,
                                activation = None,
                                name = name + '/conv2d')
        # 第二个参数很重要，normalization需要维护一个均值和一个方差，
        # 在训练过程和测试过程中，他们的值是不一样的，
        # 在训练上，均值和方差是在一个batch上计算得到的
        # 预测过程中，均值和方差是在整个数据集上，通过加权平均计算得到的
        # 所以，在训练和测试中模式是不一样的，
        # 在 train 中，设置为 True，在test中，设置为False
        bn = tf.layers.batch_normalization(conv2d,
                                           training = is_training)
        return activation(bn)




def pooling_wrapper(inputs, name):
    '''
    tf.layers.max_pooling2d 的包裹函数
    :param inputs:
    :param name:
    :return:
    '''
    return tf.layers.max_pooling2d(inputs,
                                   (2, 2),
                                   (2, 2),
                                   name = name)


# conv1:神经元 feature_map 输出图像  图像大小: 32 * 32
conv1_1 = conv_wrapper(normal_result_x_images, 'conv1_1', is_training)

conv1_2 = conv_wrapper(conv1_1, 'conv1_2', is_training)
conv1_3 = conv_wrapper(conv1_2, 'conv1_3', is_training)
# 池化层 图像输出为: 16 * 16
pooling1 = pooling_wrapper(conv1_3, 'pooling1')
conv2_1 = conv_wrapper(pooling1, 'conv2_1', is_training)
conv2_2 = conv_wrapper(conv2_1, 'conv2_2', is_training)
conv2_3 = conv_wrapper(conv2_2, 'conv2_4', is_training)
# 池化层 图像输出为 8 * 8
pooling2 = pooling_wrapper(conv2_3, 'pooling2')
conv3_1 = conv_wrapper(pooling2, 'conv3_1', is_training)
conv3_2 = conv_wrapper(conv3_1, 'conv3_2', is_training)
conv3_3 = conv_wrapper(conv3_2, 'conv3_3', is_training)
# 池化层 输出为 4 * 4 * 32
pooling3 = pooling_wrapper(conv3_3, 'pooling3')
# 展平
flatten  = tf.contrib.layers.flatten( pooling3 )
y_ = tf.layers.dense(flatten, 10)


# 使用交叉熵 设置损失函数
loss = tf.losses.sparse_softmax_cross_entropy( labels = y, logits = y_ )
# 该api,做了三件事儿 1. y_ -> softmax 2. y -> one_hot 3. loss = ylogy

# 预测值 获得的是 每一行上 最大值的 索引.注意:tf.argmax()的用法,其实和 np.argmax() 一样的
predict = tf.argmax( y_, 1 )
# 将布尔值转化为int类型,也就是 0 或者 1, 然后再和真实值进行比较. tf.equal() 返回值是布尔类型
correct_prediction = tf.equal( predict, y )
# 比如说第一行最大值索引是6,说明是第六个分类.而y正好也是6,说明预测正确



# 将上句的布尔类型 转化为 浮点类型,然后进行求平均值,实际上就是求出了准确率
accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float64) )

with tf.name_scope('train_op'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) # 将 损失函数 降到 最低


def variable_summary(var, name):
    '''
    一个变量的各种统计量,建立一个summary
    :param var: 计算summary的变量
    :param name: 指定命名空间,以防冲突
    :return:
    '''
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            # 求标准差
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean) # 均值
            tf.summary.scalar('stddev', stddev) # 标准差
            tf.summary.scalar('min', tf.reduce_min(var)) # 最小值
            tf.summary.scalar('max', tf.reduce_max(var)) # 最大值
            tf.summary.histogram('histogram', var) # 直方图 反应的是变量的分布


# 给六个卷积层添加summary
with tf.name_scope('summary'):
    variable_summary(conv1_1, 'conv1_1')
    variable_summary(conv1_2, 'conv1_2')
    variable_summary(conv2_1, 'conv2_1')
    variable_summary(conv2_2, 'conv2_2')
    variable_summary(conv3_1, 'conv3_1')
    variable_summary(conv3_2, 'conv3_2')



loss_summary = tf.summary.scalar('loss', loss)
# 'loss':<10, 1.1>, <20, 1.08>
accuracy_summary = tf.summary.scalar('accurary', accuracy)

inputs_summary = tf.summary.image('inputs_image', normal_result_x_images)

merged_summary = tf.summary.merge_all() # 将以上所有带有 summary 的变量聚合起来
merged_summary_test = tf.summary.merge([loss_summary, accuracy_summary])

# 指定文件保存路径
LOG_DIR = '.'
run_label = 'run_vgg_tensorboard'
run_dir = os.path.join(LOG_DIR, run_label)
# 判断该文件夹是否已经创建
if not os.path.exists(run_dir):
    os.mkdir(run_dir)
# 在该文件夹下创建两个文件夹,一个存放训练数据,一个存放测试数据
train_log_dir = os.path.join(run_dir, 'train')
test_log_dir = os.path.join(run_dir, 'test')
# 判断这两个文件夹是否存在
if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)
if not os.path.exists(test_log_dir):
    os.mkdir(test_log_dir)



# 初始化变量
init = tf.global_variables_initializer()

train_steps = 1000000
test_steps = 100

#　不是每一步summary都是要计算的可以定义一个范围,每过多少步计算一次
output_summary_every_steps = 100


with tf.Session() as sess:
    sess.run( init ) # 注意: 这一步必须要有!!

    # 打开一个writer,向writer中写数据
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph) # 参数2:显示计算图
    test_writer = tf.summary.FileWriter(test_log_dir)

    fixed_test_batch_data, fixed_test_batch_labels = test_data.next_batch(batch_size)

    # 开始训练
    for i in range( train_steps ):
        # 得到batch
        batch_data, batch_labels = train_data.next_batch( batch_size )


        eval_ops = [loss, accuracy, train_op]
        should_output_summary = ((i+1) % output_summary_every_steps == 0)

        if should_output_summary:
            eval_ops.append(merged_summary)


        # 获得 损失值, 准确率
        eval_val_results = sess.run( eval_ops, feed_dict={x:batch_data, y:batch_labels, is_training:True} ) # 在训练的时候，is_train 为 True
        loss_val, acc_val = eval_val_results[0:2]



        if should_output_summary:
            train_summary_str = eval_val_results[-1]
            train_writer.add_summary(train_summary_str, i+1)
            # 在 测试 时候，is_trian 为 False
            test_summary_str = sess.run([merged_summary_test],
                                        feed_dict = {x: fixed_test_batch_data,y: fixed_test_batch_labels, is_training: False} )[0]
            test_writer.add_summary(test_summary_str, i+1)



        # 每 500 次 输出一条信息
        if ( i+1 ) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % ( i+1, loss_val, acc_val ))
        # 每 5000 次 进行一次 测试
        if ( i+1 ) % 5000 == 0:
            # 获取数据集,但不随机
            test_data = CifarData( test_filename, False )
            all_test_acc_val = []
            for j in range( test_steps ):
                test_batch_data, test_batch_labels = test_data.next_batch( batch_size )
                test_acc_val = sess.run( [accuracy], feed_dict={ x:test_batch_data, y:test_batch_labels, is_training:False } )
                all_test_acc_val.append( test_acc_val )
            test_acc = np.mean( all_test_acc_val )

            print('[Test ] Step: %d, acc: %4.5f' % ( (i+1), test_acc ))