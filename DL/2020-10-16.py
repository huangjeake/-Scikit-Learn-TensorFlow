'''
gridSearchCV（网格搜索）的参数、方法及示例:
    GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。但是这个方法适合于小数据集，
    一旦数据的量级上去了，很难得出结果。这个时候就是需要动脑筋了。数据量比较大的时候可以使用一个快速调优的方法——坐标下降。
    它其实是一种贪心算法：拿当前对模型影响最大的参数调优，直到最优化；再拿下一个影响最大的参数调优，如此下去，直到所有的参数
    调整完毕。这个方法的缺点就是可能会调到局部最优而不是全局最优，但是省时间省力，巨大的优势面前，还是试一试吧，后续可以再拿
    bagging再优化。
    通常算法不够好，需要调试参数时必不可少。比如SVM的惩罚因子C，核函数kernel，gamma参数等，对于不同的数据使用不同的参数，结果
    效果可能差1-5个点，sklearn为我们提供专门调试参数的函数grid_search。

重要性分析：线性回归model.coef_
            或者决策树 随机森林 XGBoost model.feature_importances_ 返回对应特征的系数，系数总和为1

lesson2 week1
'''