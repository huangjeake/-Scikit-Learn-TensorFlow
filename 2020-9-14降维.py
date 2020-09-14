'''

降维：三种数据降维技术：PCA、Kernal PCA以及LLE
    训练集的维度越高，数据越稀疏，过度拟合的风险就越大。
    理论上来说，通过增大训练集，使训练实例达到足够的密度，是可以解开维度的诅咒的。然而不幸的是，实践中，要达到给定密度所
    需要的训练实例数量随着维度增加呈指数式上升。仅仅100个特征下（远小于MNIST问题），要让所有训练实例（假设在所有维度上平
    均分布）之间的平均距离小于0.1，你需要的训练实例数量就比可观察宇宙中的原子数量还要多。

降维的方法：投影和流形学习
    投影：将高维空间的数据投影到低维空间
    流行学习包含的假设：如果能用低维空间的流形表示，手头的任务（例如分类或者回归）将变得更简单
    在训练模型之前降低训练集的维度，肯定可以加快训练速度，但这并不总是会导致更好或更简单的解决方案，它取决于数据集。


PCA：主成份分析
    先是识别出最接近数据的超平面，然后将数据投影其上。保留最大差异性，减少信息丢失，计算数据集与超平面的最小均方距离
    也可以得出该结果


np.linalg.inv矩阵求逆
np.linalg.eigh矩阵特征向量
np.linalg.svd奇异值获取主成份




'''