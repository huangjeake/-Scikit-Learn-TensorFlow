'''
提升法：将多个弱学习器结合成一个强学习器的任意集成方法。
    分为两类：adaboost自适应提升法 梯度提升
    1、利用错题提升学习效率
    做正确的题，下次少做点，反正都会了。
    做错的题，下次多做点，集中在错题上。
    随着学习的深入，做错的题会越来越少。
    2、合理跨界提高盈利
    苹果公司，软硬结合，占据了大部分的手机市场利润，两个领域的知识结合起来产生新收益。


adaboost:要构建一个AdaBoost分类器，首先需要训练一个基础分类器（比如决策树），用它对训练集进行预测。然后对错误分类的训练
    实例增加其相对权重，接着，使用这个最新的权重对第二个分类器进行训练，然后再次对训练集进行预测，继续更新权重，并不断循环向前
    预测器权重 = nlog((1-rj)/rj)  n是学习率默认是1，rj是加权误差率，预测越准确，权重越大；随机预测rj = 0.5 权重是1；rj < 0.5,权重为负数
    权重更新规则：更新后的权重训练一个新的预测器，然后重复整个过程（计算新预测器的权重，更新实例权重，然后对另一个预测器进行
    训练，等等）。当到达所需数量的预测器，或得到完美的预测器时，算法停止。
    预测的时候，AdaBoost就是简单地计算所有预测器的预测结果，并使用预测器权重αj对它们进行加权。最后，得到大多数加权投票的类别就是
    预测器给出的预测类别
    建议：
        如果你的AdaBoost集成过度拟合训练集，你可以试试减少估算器数量，或是提高基础估算器的正则化程度。

梯度提升：Gradient Boosting
    不是像AdaBoost那样在每个迭代中调整实例权重，而是让新的预测器针对前一个预测器的残差进行拟合。
    from sklearn.tree import DecisionTreeRegressor
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(X, y)
    y2 = y - tree_reg1.predict(X)
    tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg2.fit(X, y2)
    y3 = y2 - tree_reg2.predict(X)
    tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg3.fit(X, y3)
    X_new = np.array([[0.8]])
    y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    简单方法：from sklearn.ensemble import  GradientBoostingRegressor 与RandomForestRegressor类似，它具有控制
    决策树生长的超参数（例如max_depth、min_samples_leaf等），以及控制集成训练的超参数，例如树的数量（n_estimators）。

    要找到树的最佳数量，可以使用早期停止法（参见第4章）。简单的实现方法就是使用staged_predict（）方法：它在训练的每个阶段
    （一棵树时，两棵树时，等等）都对集成的预测返回一个迭代器。
    import numpy as np
    from sklearn.ensemble import  GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)
    gbrt =  GradientBoostingRegressor(max_depth=2,n_estimators=120, random_state=42)
    gbrt.fit(X_train, y_train)
    errors = [mean_squared_error(y_val,y_pred) for y_pred in gbrt.staged_predict(X_val)]
    bst_n_estimators = np.argmin(errors)
    gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators, random_state=42)
    gbrt_best.fit(X_train, y_train)

    方法一：通过训练大量的树，通过mean_squared_error　np.argmin回头找出最优的训练器个数
    min_val_error = float("inf")
    gbrt = GradientBoostingRegressor(max_depth=2,random_state=42,warm_start=True)
    error_going_up = 0
    for n_estimators in range(1, 121):
        gbrt.n_estimators = n_estimators
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_error = mean_squared_error(y_val,y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                break
    print(gbrt.n_estimators)
    方法二：通过warm_start=True，当fit（）方法被调用时，Scikit-Learn会保留现有的树，从而允许增量训练
    当训练得出的误差连续５次比最小误差小，即可得出最优解


注释：还有一种集成学习的方法叫做堆叠，sklearn并不支持

'''
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split