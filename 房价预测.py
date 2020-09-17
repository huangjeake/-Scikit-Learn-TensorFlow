from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error


def model():
    """
    正规方程
    获取基本数据
    数据基本处理
    特征工程
    机器学习
    模型评估
    :return:
    """
    data = load_boston()
    print(data)

    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=22)

    #特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    #机器学习
    estimator = LinearRegression()
    estimator.fit(x_train,y_train)

    #模型评估
    print('目标与测试：\n', estimator.predict(x_test))
    print('模型系数：\n', estimator.coef_)
    print('模型偏置量：\n', estimator.intercept_)

    #模型评价
    print('模型方差:\n', mean_squared_error(y_test, estimator.predict(x_test)))
    print('准确率:\n', estimator.score(x_test, y_test))


def mode2():
    """
    梯度下降法
    获取基本数据
    数据基本处理
    特征工程
    机器学习
    模型评估
    :return:
    """
    data = load_boston()
    print(data)

    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=22)

    #特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    #机器学习
    estimator = SGDRegressor()
    estimator.fit(x_train,y_train)

    #模型评估
    print('目标与测试：\n', estimator.predict(x_test))
    print('模型系数：\n', estimator.coef_)
    print('模型偏置量：\n', estimator.intercept_)

    #模型评价
    print('模型方差:\n', mean_squared_error(y_test, estimator.predict(x_test)))
    print('准确率:\n', estimator.score(x_test, y_test))

def mode3():
    """
    岭回归
    获取基本数据
    数据基本处理
    特征工程
    机器学习
    模型评估
    :return:
    """
    data = load_boston()
    print(data)

    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=22)

    #特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    #机器学习
    estimator = Ridge(alpha=1)
    # estimator = RidgeCV(alphas=(0.1, 1, 10))#可以进行交叉验证
    estimator.fit(x_train,y_train)

    #模型评估
    print('目标与测试：\n', estimator.predict(x_test))
    print('模型系数：\n', estimator.coef_)
    print('模型偏置量：\n', estimator.intercept_)

    #模型评价
    print('模型方差:\n', mean_squared_error(y_test, estimator.predict(x_test)))
    print('准确率:\n', estimator.score(x_test, y_test))


if __name__ == '__main__':
    model()
    mode2()
    mode3()