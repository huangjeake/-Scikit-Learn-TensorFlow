'''
机器学习的分类： 监督学习 无监督学习 半监督式学习 强化学习
监督式学习的算法：K临近算法 线性回归 逻辑回归 支持向量机 决策树和随机森林 神经网络
无监督式学习：
    聚类算法： K均值算法 分层聚类分析 最大期望算法
    可视化和降维：PCA 核主成分分析 局部线性嵌入 t-分布随机临近嵌入
    关联规则学习：Apriori  Echat
过拟合方法： 简化模型（减少属性数量或者正则化） 收集更多的训练数据 减少训练数据中的噪声（修复数据错误和消除异常值）
拟合不足的方法：选择一个带有更多参数，更强大的模型
               给学习算法提供更好的特征集
               减少模型中的约束（比如减少正则化超参数）
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    # pivot 相当于reshape
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    # 重命名
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    # 设置索引
    gdp_per_capita.set_index("Country", inplace=True)
    # 合并
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    # 排序
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
oecd_bli = pd.read_csv('./handson-ml/datasets/lifesat/oecd_bli_2015.csv', thousands=',')
# thousands千位分隔符
# delimiter 以什么分割
# na_values 将n/a设置为NAN
gdp_per_capita = pd.read_csv('./handson-ml/datasets/lifesat/gdp_per_capita.csv', thousands=',',\
                             delimiter='\t',encoding='latin1', na_values='n/a')


# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
# 合并 按列合并 np._r按行合并
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]