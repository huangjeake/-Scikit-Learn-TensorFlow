'''
支持向量机可以用于分类，回归，异常值检测任务；也被称为最大间隔分类
SVM对特征的缩放非常敏感，如图5-2所示，在左图中，垂直刻度比水平刻度大得多，因此可能的最宽的街道接近于水平。在特
征缩放（例如使用Scikit-Learn的StandardScaler）后，决策边界看起来好很多

软间隔分类：
    硬间隔分类对于异常值非常敏感
    svm中通过超参数C来控制这个平衡，C值越小则街道越宽，违例间隔也会越多
    C值越大，街道越窄

np.concatenate拼接
plt.axhline(y=0)绘制垂直于x轴的线
plt.gca移动坐标轴
plt.contourf绘制等高线
np.argwhere返回条件的索引


from sklearn.svm import LinearSVC,SVC
SVC(kernel="poly", degree=10, coef0=100, C=5)参数degree控制多次项
阶数，cefo控制高阶影响的大小

高斯RBF核函数：
    SVC(kernel="rbf", degree=10, coef0=100, C=5)适用于训练样本较少的情况
    线性支持向量机比支持向量机核函数快，适用于大样本



'''
from sklearn.svm import LinearSVC,SVC
#
# LinearSVC
from numpy import  *
for i, j in ((1, 2),(2, 3)):
    print(i, j)
print(esp_1)