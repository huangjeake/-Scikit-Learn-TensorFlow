'''
决策树可以做分类和回归：
决策树的特征之一就是他们需要的数据准备工作非常少，特别是，完全不需要进行特征缩放或者集中

cart 分类与回归树
    是一种贪婪算法，从顶层开始搜索最优分裂，然后每层重复这个过程，几层分裂之后，它并不会检视这个分裂的不纯度是否为可能的
    最低值。贪婪算法通常会产生一个相当不错的解，但是不能保证是最优解。

决策树默认使用基尼不纯度来测量，也可以使用信息嫡的方式测量
基尼不纯度  gini = 1 - sum(pk**2) p第i个节点上 k训练实例占比
信息嫡：h = -sum(klogk) k训练实例占比

决策树容易导致过拟合需要正则化超参数，min_samples_split分裂前节点必须有的最小样本数
min_samples_leaf叶子点必须有的最小样本数量，min_weight_fraction_leaf加权实例总数的占比
max_leaf_nodes 最大叶节点数量

回归：
    与分类不一样的地方。分类最小化不纯度，回归最小化成本函数；也可以通过正则化超参数防止过拟合

不稳定性：
    决策树的主要问题是它们对训练数据中的小变化非常敏感。
'''