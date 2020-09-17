'''
回归任务--预测值
分类任务--预测类
准确率无法成为分类器的首要性能指标，123，非3的准确是66.7% 这里需要使用混淆矩阵

精度和召回率：
    召回率=真正例/（真正例+假反例）。也就是正确判为恐怖分子占实际所有恐怖分子的比例。
    精度=真正例/（真正例+假正例）：也就是在所有判为恐怖分子中，真正的恐怖分子的比例。
    随着精度的增加，召回率会降低，反之亦然。
精度和召回率为纵坐标，阈值是横坐标
梯度下降交叉验证method="decision_function"，用decision_function()方法预测决策的分数
随机森林交叉验证method="predict_proba" 方法预测决策的分数
判定一个二分类器的好坏：  精度--召回率曲线   ROC曲线
分析混淆矩阵confusion_matrix可以帮助深入了解如何改进分类器

分类任务指标混淆混镇，准确率不可靠；精度和召回率 ROC曲线
分类器有一对一，一对多；比较分类器需要将混淆矩阵可视化，明亮的地方就是分类器效果差的
多标签分类器：评估多标签分类器取决于项目
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.datasets import fetch_openml
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]
mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
print(mnist)

import numpy as np
a = np.random.randn(20)
# np.reshape