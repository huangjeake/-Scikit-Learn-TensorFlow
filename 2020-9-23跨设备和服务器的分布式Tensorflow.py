'''

分布式TF中，TF需要建立一个集群，然后在集群中建立两个job，一个是ps job，负责参数初始化，参数更新，一个job下面可以有多个task
（有多个task，说明有多台机器，或者GPU负责参数初始化，更新）。一个是woker job，负责计算图的运算，计算梯度，一个worker job下面
也可以有很多个task（有多个task，说明有多台机器，或者GPU负责运行计算图）


跨多参数服务器分片变量:
    replica_device_setter（）方法，它以循环方式在所有"ps"任务之配置变量

图内分布式中，计算图只有一个，需要一个中心节点分配计算任务并更新参数，由于中心节点的存在，中心节点容易成为瓶颈.
图间分布式中，计算图有多个，但是不同计算图的相同变量通过tf.train.replica_device_setter函数放到同一个服务器上，这种情况下，
各个计算图相互独立（参数只有一份，计算图有多个），并行度更高，适合异步更新.一般采用图间分布式。


使用协调器和QueueRunner的多线程读取器：
    coord = tf.train.Coordinator()创建Coordinator对象
    coord.request_stop()调用request_stop（）方法来停止每个线程
    coord.join(list_of_threads)调用join（）方法来等待所有线程结束


带宽饱和:将数据从GPU RAM（并且可能跨网络）传入传出所花的时间超过了通过分割运算负载所获得的增速





tf.device()指定tensorflow运行的GPU或CPU设备  tf.device('/gpu:1') 指定Session在第二块GPU上运行
tf.train.ClusterSpec()：创建一个集群对象
tf.train.Server()：在这个集群上面创建一个服务器，根据实际情况，可以是参数服务器，也可以是计算服务器
tf.train.Supervisor()：创建一个监视器，就是用来监控训练过程的，个人感觉主要就是方便恢复模型训练，其logdir参数为训练日志目录，如果
里面有模型，则直接恢复训练。所以如果想重新训练，需要删除这个目录。
sv.managed_session()：启动Session，相比于其他启动Session的方式，多了一些功能
tf.train.replica_device_setter可以自动把Graph中的Variables放到ps上，而同时将Graph的计算部分放置在当前worker上
tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])用TensorFlow的CSV解析器从当前行中取值。当缺少一个字段（这个例子中第三
个训练实例的x2特征）时会使用默认值，并且它们还用来确认每一字段的类型（本例中为两个浮点型和1个整型）。

tf.FIFOQueue根据先进先出（FIFO）的原则创建一个队列
tf.TextLineReader输出由换行符分隔的文件行的读取器

tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2,dtypes=[tf.float32, tf.int32], shapes=[[2],[]],name="instance_q",
 shared_name="shared_instance_q")
capacity：一个整数。可以存储在此队列中的元素数的上限。
min_after_dequeue:指定在a dequeue或dequeue_many操作完成后将保留在队列中的最小元素数，以确保最小程度的元素混合。通过阻塞那些
操作直到队列中有足够的元素来保持不变。
dtypes：DType对象列表的长度dtypes必须等于每个队列元素中的张量的数量。
shapes：（可选。）TensorShape与dtypes或相同长度的完全定义对象的列表None。
names：（可选。）一个字符串列表，用于命名队列中与dtypes或相同长度的组件None。如果指定了出队方法，则返回以名称为键的字典。
seed：Python整数。用于创建随机种子。请参阅tf.set_random_seed行为。
shared_name：（可选。）如果为非空，则该队列将以给定名称在多个会话中共享。
name：队列操作的可选名称。

tf.PaddingFIFOQueue 以固定长度批量出列的队列
tf.PriorityQueue 带优先级出列的队列
tf.ConfigProto()主要的作用是配置tf.Session的运算方式，比如gpu运算或者cpu运算
tf.data.Dataset.from_tensor_slices 把给定的元组、列表和张量等数据进行特征切片。切片的范围是从最外层维度开始的。如果有多个
特征进行组合，那么一次切片是把每个组合的最外维度的数据切开，分成一组一组的

'''
# from sklearn.datasets import load_iris
# iris  = load_iris()
# print(iris)
import  numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def read_and_push_instance(filename_queue, instance_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    x1, x2, target = tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])
    features = tf.stack([x1, x2])
    enqueue_instance = instance_queue.enqueue([features, target])
    return enqueue_instance

filename_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
filename = tf.placeholder(tf.string)
enqueue_filename = filename_queue.enqueue([filename])
close_filename_queue = filename_queue.close()

instance_queue = tf.RandomShuffleQueue(
    capacity=10, min_after_dequeue=2,
    dtypes=[tf.float32, tf.int32], shapes=[[2],[]],
    name="instance_q", shared_name="shared_instance_q")

minibatch_instances, minibatch_targets = instance_queue.dequeue_up_to(2)

read_and_enqueue_ops = [read_and_push_instance(filename_queue, instance_queue) for i in range(5)]
queue_runner = tf.train.QueueRunner(instance_queue, read_and_enqueue_ops)

with tf.Session() as sess:
    sess.run(enqueue_filename, feed_dict={filename: "my_test.csv"})
    sess.run(close_filename_queue)
    coord = tf.train.Coordinator()
    enqueue_threads = queue_runner.create_threads(sess, coord=coord, start=True)
    try:
        while True:
            print(sess.run([minibatch_instances, minibatch_targets]))
    except tf.errors.OutOfRangeError as ex:
        print("No more training instances")
