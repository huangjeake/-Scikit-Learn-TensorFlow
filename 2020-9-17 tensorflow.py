''''
tensorflow的运行规律，就是步骤了：
1， 创建tensorflow变量，初始变量，用于执行的的
2， 设置操作的配置
3， 初始化tensorflow 例子
4， 创建tensorflow session （session是执行引擎）
5， 执行session 即运行你的例子

tensorflow包含两部分：
    计算图和执行图；计算图包括变量的定义等；开始会话，执行图；

张量就是数组或者列表；
变量的定义：
    import tensorflow as tf
    x = tf.Variable(value, name='x')

创建会话的三种方式：使用with不需要关闭会话。
with  tf.Session() as ss:
    x.initializer.run()# 等价于tf.session.run(initializer)
    y.initializer.run()# tf.session.run(initializer)
    result = f.eval()

init = tf.global_variables_initializer() # prepare an init node
with tf.Session() as sess:
init.run() # actually initialize all the variables 初始化所有变量
result = f.eval()

sess = tf.InteractiveSession()# 互动会话
int.run()
result2 = f.eval()
sess.close()

管理图：
    创建的所有节点都会自动添加到默认图上，如果管理多个互不依赖的图
    可以创建一个新的图，然后用with块临时将他设置为默认图

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:#每次执行图都会计算z,y的值
    # print(y.eval()) # 10
    # print(z.eval()) # 15

     # z,y = sess.run([z, y])# 在一次图的执行中就完成z,y的求值
     # print(z, y)

在单进程的TensorFlow中，即使它们共享同一个计算图，多
个会话之间仍然互相隔离，不共享任何状态（每个会话对每个变量都
有自己的拷贝）。对于分布式TensorFlow（见第12章），变量值保存
在每个服务器上，而不是会话中，所以多个会话可以共享同一变量


使用优化器：
    梯度下降优化器：
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)
    动量优化器：
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    training_op = optimizer.minimize(mse)

保存模型：
    创建一个Saver节点，然后在执行期，调用save（）方法，并传入一个会话和检查点文件的路径即可保存模型：
    saver = tf.train.Saver()
    在执行的时候：save = saver.save(session,path)#传入会话和路径即可

导入模型：
    恢复模型同样简单：与之前一样，在构造期末尾创建一个Saver节点，不过在执行期开始的时候，不是用init节点来初始化变量，而
    是调用Saver对象上的restore（）方法：
    with tf.Session() as sess:
        saver.restore(sess, "/tmp/my_model_final.ckpt")

命名作用域：
    with tf.name_scope("loss") as scope:
         error = y_pred - y
         mse = tf.reduce_mean(tf.square(error), name="mse")
    将相关节点通过命名作用域分组

共享变量：
    def relu(X):
        with tf.variable_scope("relu", reuse=True):
            threshold = tf.get_variable("threshold")
            w_shape = int(X.get_shape()[1]), 1                          # not shown
            w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
            b = tf.Variable(0.0, name="bias")                           # not shown
            z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
            return tf.maximum(z, threshold, name="max")

    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    with tf.variable_scope("relu"):
        threshold = tf.get_variable("threshold", shape=(),
                                    initializer=tf.constant_initializer(0.0))
    relus = [relu(X) for relu_index in range(5)]
    output = tf.add_n(relus, name="output")

XT = tf.transpose(X)转置等价于array.T
tf.matmul() 和tf.multiply() 的区别：
    tf.matmul() 是两个矩阵的相乘；tf.multiply()是矩阵内部数字之间的乘法，数字和数字之间的乘法，数字和矩阵之间的乘法
tf.matrix_diag生成对角矩阵
tf.matrix_inverse矩阵的逆
tf.random_uniform 产生多维数组 与numpy.random.uniform一致
tf.reduce_mean计算平均值
tf.assign(theta, theta - learning_rate * gradients)将后面的参数复制给前面的参数
gradients = tf.gradients(mse, [theta])前面的对后面的参数求导
tf.placeholder(tf.float32, shape=(None, 3))创建一个占位符节点，并且制定张量的形状，多行3列
tf.summary.scalar用来显示标量信息 一般在画loss accuary图的时候会用到
tf.train.import_meta_graph用来加载meta文件中的图,以及图上定义的结点参数包括权重偏置项等需要训练的参数,也包括训练过程生成的中间参数
saver.restore()只是加载了参数，配置项等
tf.add_n([p1, p2, p3....])函数是实现一个列表的元素的相加









'''