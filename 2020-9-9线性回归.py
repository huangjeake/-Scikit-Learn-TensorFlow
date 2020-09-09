'''
梯度下降方法：使损失函数最小化。学习率即是设定的步长，学习率过低需要大量迭代，学习率过高可能越过最低点

线性回归模型：
    即便是乱走，梯度下降都可以趋近到全局最小值（只要等待时间足够长，学习率也不是太高）。

批量梯度下降：每一步都使用整批训练数据。因此，面对非常庞大的训练集时，算法会变得极慢

随机梯度下降：它比批量梯度下降要不规则得多。成本函数将不再是缓缓降低直到抵达最小值，而是不断上上下下，但是从整体来看，还是在慢慢下降。随着时间推移，最终会非常
    接近最小值，但是即使它到达了最小值，依旧还会持续反弹，永远不会停止（见图4-9）。所以算法停下来的参数值肯定是足够好的，但
    不是最优的。随机梯度下降其实可以帮助算法跳出局部最小值，所以相比批量梯度下降，它对找到全局最小值更有优势。
from sklearn.linear_model import SGDClassifier
SGDRegressor(max_iter=50, tol=-np.infty, penalty=None, eta0=0.1, random_state=42)#max_iter最大迭代 penalty是否正则化 eta0学习率 random_state随机种子

多项式回归模型预测：
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree=2, include_bias=False)degree=3是阶数

判断模型是否欠拟合或者过拟合：
    1.通过交叉验证来评估模型
    from sklearn.model_selection import cross_val_score
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
... scoring="neg_mean_squared_error", cv=10)
    2.学习曲线
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    def plot_learning_curves(model,X,y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
        train_errors, val_errors = [], []
        for m in range(1,len(X_train)):
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
            val_errors.append(mean_squared_error(y_val,y_val_predict))
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
        plt.legend(loc="upper right", fontsize=14)   # not shown in the book
        plt.xlabel("Training set size", fontsize=14) # not shown
        plt.ylabel("RMSE", fontsize=14)              # not shown

    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)
    plt.axis([0, 80, 0, 3])                         # not shown in the book
    plt.savefig("underfitting_learning_curves_plot")   # not shown
    plt.show()

正则线性模型：(减少过度拟合)
    岭回归：正则化l2。将高次项系数变小，降低高次项对模型的影响
    lasso回归：正则化l1.将高次项系数变o，降低高次项对模型的影响
    弹性网络：介于岭回归和lasso回归之间

    大多数情况下，你应该避免使用纯线性回归。岭回归是个不错的默认选择，但是如果你觉得实际用到的特征只有少数几个，那就应该
    更倾向于Lasso回归或是弹性网络，因为它们会将无用特征的权重降为零。一般而言，弹性网络优于Lasso回归，因为当特征数量超过训
    练实例数量，又或者是几个特征强相关时，Lasso回归的表现可能非常不稳定。


'''
from sklearn.linear_model import SGDRegressor
import numpy as np
from sklearn.model_selection import cross_val_score
sgc_estimator = SGDRegressor(max_iter=100,)
print(np.random.permutation(10))
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.plot(np.sqrt(mean_squared_error), 'b+', labe)