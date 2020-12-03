from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, RidgeCV  # 线性 随机梯度 岭回归
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.model_selection import train_test_split, GridSearchCV  # 训练测试集 网格交叉
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, classification_report  # 标准方差 评估报告
import joblib  # 保存 加载模型  方法二：python自带的pickle 方法三：tf.train.saver()
import pandas as pd
from sklearn.metrics import roc_auc_score  # AUC分数
from sklearn.feature_extraction import DictVectorizer  # 字典特征提取
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # 决策树 决策树报告
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier  # 随机森林
from sklearn.cluster import KMeans  # Kmeans聚类算法


def model():
    """
    获取基本数据
    数据基本处理
    特征工程
    机器学习
    模型评估
    :return:
    sklearn.feature_extraction.DictVectorizer中文特征提取
    sklearn.feature_extraction.text.CountVectorizer文本特征提取
    中文支持 jieba
    """
    data = pd.read_csv(r'C:\Users\Administrator\Desktop\资料\上交所主板.csv')

    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=22)

    # 特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # #机器学习
    # estimator = Ridge(alpha=1)
    # # estimator = RidgeCV(alphas=(0.1, 1, 10))#可以进行交叉验证
    # estimator = GridSearchCV(estimator, param_grid=param_dict, cv=5)#可以进行交叉验证
    # estimator.fit(x_train,y_train)

    # #模型保存
    # joblib.dump(estimator, './boston.pkl')

    # 加载模型
    estimator = joblib.load('./boston.pkl')
    y_predict = estimator.predict(x_test)
    # 模型评估
    print('目标与测试：\n', estimator.predict(x_test))
    print('模型系数：\n', estimator.coef_)
    print('模型偏置量：\n', estimator.intercept_)

    # 模型评价
    print('模型方差:\n', mean_squared_error(y_test, estimator.predict(x_test)))
    print('准确率:\n', estimator.score(x_test, y_test))
    # AUC指标 0.5-1之间 越接近1越好
    print("AUC指标：", roc_auc_score(y_test, y_predict))
    # 评估报告
    ret = classification_report(y_test, y_predict)
    print(ret)

    # 保存树的结构到dot文件
    # export_graphviz(estimator, out_file="./data/tree.dot")#http://webgraphviz.com/显示树状结构


model()
