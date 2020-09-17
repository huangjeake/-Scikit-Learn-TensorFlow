from sklearn.datasets import load_iris,fetch_20newsgroups
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
#获取小数据集
iris = load_iris()
# print(iris)

#获取大数据集
# news = fetch_20newsgroups()
# print(news)

# print("鸢尾花数据集的返回值：\n", iris)
# # # 返回值是一个继承自字典的Bench
# print("鸢尾花的特征值:\n", iris["data"])
# print("鸢尾花的目标值：\n", iris.target)
# print("鸢尾花特征的名字：\n", iris.feature_names)
# print("鸢尾花目标值的名字：\n", iris.target_names)
# print("鸢尾花的描述：\n", iris.DESCR)

#数据集的可视化
iris_d = pd.DataFrame(iris['data'], columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris_d['target'] = iris.target
print(iris_d)

# def plot_iris(data,col1,col2):
#     sns.lmplot(x=col1,y=col2,data=data,hue='target',fit_reg=False)
#     plt.show()
#     plt.xlabel(col1)
#     plt.ylabel(col2)
#     plt.title('鸢尾花种类分布图')
#
#
# plot_iris(iris_d,'Petal_Length', 'Petal_Width')

# 数据集的划分
# x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
# print('特征值训练集',x_train,x_train.shape)
# print('特征值测试集',x_test,x_test.shape)
# print('目标值训练集',y_train,y_train.shape)
# print('目标值测试集',y_test,y_test.shape)