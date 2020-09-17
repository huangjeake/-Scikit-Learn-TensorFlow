'''
感知器：
    感知器是人工神经网络中的一种典型结构， 它的主要的特点是结构简单，对所能解决的问题 存在着收敛算法，
    并能从数学上严格证明，从而对神经网络研究起了重要的推动作用
    和逻辑回归分类器相反，感知器不输出某个类概率。它只能根据一个固定的阈值来做预测 单个感知器无法解决异或分类问题
    异或分类:将数字转化为二进制的，不同的为0，相同的部分为1

反向传播算法：
    反向传播算法先做一次预测（正向过程），度量误差，然后反向的遍历每个层次来度量每个连接的误差贡献度（反向过程），最
    后再微调每个连接的权重来降低误差（梯度下降）。


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

'''
import tensorflow as tf
a = tf.estimator.inputs.numpy_input_fn()