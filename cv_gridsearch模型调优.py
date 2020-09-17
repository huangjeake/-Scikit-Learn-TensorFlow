from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#获取数据
iris = load_iris()

#数据特征处理
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)

#特征工程,标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#机器学习。模型训练
estimator = KNeighborsClassifier()
#交叉验证，模型调优
param_dict = {'n_neighbors':[1,3,5,7,9]}
estimator = GridSearchCV(estimator, param_grid=param_dict,cv=5)

estimator.fit(x_train, y_train)
#模型预测
# 方法1：比对真实值和预测值
y_predict = estimator.predict(x_test)
# print("预测结果为:\n", y_predict)
# print("比对真实值和预测值：\n", y_predict == y_test)
# # 方法2：直接计算准确率
# score = estimator.score(x_test, y_test)
# print("准确率为：\n", score)

#交叉验证，网格搜索属性
print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
print("最好的参数模型：\n", estimator.best_estimator_)
print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)