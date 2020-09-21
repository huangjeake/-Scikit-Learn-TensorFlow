
'''
安装了英伟达显卡可以调用nvidia-smi命令来检查CUDA是否安装成功
管理GPU：
    设置环境变量
    $ CUDA_VISIBLE_DEVICES=0,1 python3 program_1.py
    # and in another terminal:
    $ CUDA_VISIBLE_DEVICES=3,2 python3 program_2.py
    每个程序本身就有两个GPU
    TensorFlow只占每个GPU内存的40%，必须创建ConfigProto对象，
    将gpu_options.per_process_gpu_memory_fraction选项设置为0.4

设备上的操作：
    "/cpu：0"设备聚合了一个多CPU系统的所有CPU。当前没有什么方法可以固定节点到特定的CPU或者只使用所有CPU的一部分

TensorFlow针对整数变量就没有GPU内核



'''
import tensorflow as tf
# tf.compat.v1.Session() tensorflow的导入用这种方式
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.Session()

with tf.device('/cpu:0'):
    a = tf.Variable(3.0)
    b = tf.constant(4.0)
c = a + b

