# # //导入饼图Pie
# from pyecharts.charts import Pie
# columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# # //设置数据
# data1 = [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3]
# data2 = [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3]
# # //设置主标题与副标题，标题设置居中，设置宽度为900
# pie = Pie()
# # //加入数据，设置坐标位置为【25，50】，上方的colums选项取消显示
# pie.add("降水量", columns, data1 ,center=[25,50])
# # //加入数据，设置坐标位置为【75，50】，上方的colums选项取消显示，显示label标签
# pie.add("蒸发量", columns, data2 ,center=[75,50])
# # //保存图表
# pie.render()

from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from pandas.plotting import scatter_matrix # 散点图矩阵
# 导入必要的数据集和算法


print(__doc__)

# 在stdout上显示进度日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# 图像数组以找到形状（绘图）
n_samples, h, w = lfw_people.images.shape
print(lfw_people.images.shape, lfw_people.data)

# 对于机器学习，我们直接使用2个数据（由于该模型忽略了相对像素位置信息）
X = lfw_people.data
n_features = X.shape[1]

# 预测的标签是该人的身份
y = lfw_people.target
# y为特征脸的标签
target_names = lfw_people.target_names
# 设置标签的名字
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# 分为测试集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
# 测试集大小为全部数据集的25%

n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
# n_components：这个参数可以帮我们指定希望PCA降维后的特征维度数目
# whiten： 白化。所谓白化，就是对降维后的数据的每个特征进行标准化，让方差都为1。
# svd_solver：即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
# 设置PCA降维
print("done in %0.3fs" % (time() - t0))
# 输出总耗时

eigenfaces = pca.components_.reshape((n_components, h, w))
# 将图像转换为矩阵向量
print(h, w, eigenfaces)

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
# 在测试集上PCA降维
X_test_pca = pca.transform(X_test)
# 在数据集上PCA降维
print("done in %0.3fs" % (time() - t0))

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# 绘制测试结果的一部分

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# 绘制特征脸

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
