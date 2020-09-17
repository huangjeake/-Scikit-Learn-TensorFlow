from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
x = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
# 实例化API
estimator = KNeighborsClassifier(n_neighbors=3)
# 使用fit方法进行训练,x是特征值，y是目标值
estimator.fit(x, y)

print(estimator.predict([[-1]]))

from sklearn.neighbors import KNeighborsClassifier
x = [[1], [3], [2], [5]]
y = [-1, 1, 0, 3]
es = KNeighborsClassifier(n_neighbors=5)
es.fit(x, y)
print(estimator.predict([[2]]))
