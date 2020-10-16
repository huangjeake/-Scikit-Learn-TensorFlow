'''

lesson1 week3
浅层神经网络：


卷积神经网络：
    激活函数：
    ①不要用sigmoid！不要用sigmoid！不要用sigmoid！#因为不能快速收敛
    ② 首先试RELU，因为快，但要小心点
    ③ 如果2失效，请用Leaky ReLU或者Maxout
    ④ 某些情况下tanh倒是有不错的结果，但是很少
    池化层：
    1.池化层夹在连续的卷积层中间， 用于压缩数据和参数的量，减小过拟合。简而言之，如果输入是图像的话，那么池化层的最主要作用就是压缩图像。
    特征不变性，也就是我们在图像处理中经常提到的特征的尺度不变性，池化操作就是图像的resize，平时一张狗的图像被缩小了一倍我们还能认出这是
    一张狗的照片，这说明这张图像中仍保留着狗最重要的特征，我们一看就能判断图像中画的是一只狗，图像压缩时去掉的信息只是一些无关紧要的信息，
    而留下的信息则是具有尺度不变性的特征，是最能表达图像的特征。
    2.特征降维，我们知道一幅图像含有的信息是很大的，特征也很多，但是有些信息对于我们做图像任务时没有太多用途或者有重复，我们可以把这类冗余
    信息去除，把最重要的特征抽取出来，这也是池化操作的一大作用。
    3.在一定程度上防止过拟合，更方便优化。
    池化层用的方法有Max pooling 和 average pooling，而实际用的较多的是Max pooling。
    这里就说一下Max pooling，其实思想非常简单，每次池化选出最大的数作为输出矩阵的相应元素的值
    一般CNN结构依次为：
    1.INPUT
    2.[[CONV -> RELU]N -> POOL?]M
    3.[FC -> RELU]*K
    4.FC
    卷积神经网络之优缺点
    优点
    　　•共享卷积核，对高维数据处理无压力
    　　•无需手动选取特征，训练好权重，即得特征分类效果好
    缺点
    　　•需要调参，需要大样本量，训练最好要GPU
    　　•物理含义不明确（也就说，我们并不知道没个卷积层到底提取到的是什么特征，而且神经网络本身就是一种难以解释的“黑箱模型”）

    卷积神经网络之典型CNN
    •LeNet，这是最早用于数字识别的CNN
    •AlexNet， 2012 ILSVRC比赛远超第2名的CNN，比
    •LeNet更深，用多层小卷积层叠加替换单大卷积层。
    •ZF Net， 2013 ILSVRC比赛冠军
    •GoogLeNet， 2014 ILSVRC比赛冠军
    •VGGNet， 2014 ILSVRC比赛中的模型，图像识别略差于GoogLeNet，但是在很多图像转化学习问题(比如object detection)上效果奇好

    卷积神经网络的常用框架
    Caffe
    　•源于Berkeley的主流CV工具包，支持C++,python,matlab
    　•Model Zoo中有大量预训练好的模型供使用
    Torch
    　•Facebook用的卷积神经网络工具包
    　•通过时域卷积的本地接口，使用非常直观
    　•定义新网络层简单
    TensorFlow
    　•Google的深度学习框架
    　•TensorBoard可视化很方便
    　•数据和模型并行化好，速度快











'''