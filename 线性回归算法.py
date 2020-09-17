from sklearn.linear_model import LinearRegression
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge

x = [[80, 86],
[82, 80],
[85, 78],
[90, 90],
[86, 82],
[82, 90],
[78, 80],
[92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

estimator = LinearRegression()
estimator.fit(x,y)
print('线性回归系数是：\n',estimator.coef_)
print('线性回归的值是：\n',estimator.predict([[80,90]]))