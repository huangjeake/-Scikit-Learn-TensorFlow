'''
流行的开放数据存储库：
    UC irvine Machine Learning Repository
    Kaggle datasets
    Amazon's AWS datasets
    Wikipedia's list of Machine Learning datasets
    Quora.com question
    Datasets subreddit
    https://dataportals.org
    https://opendatamonitor.eu
    https://quandl.com
sklearn里面随机种子确保每次去的数据都是一样的；不设置会导致每次训练的数据都不一样，有可能将全部数据取出来
方法一:保存训练和数据；方法二：随机种子 np.random.seed(42) np.random.permutation()
相关性分析：corr()  from pandas.plotting import scatter_matrix
数据清理：
    放弃这些相应的地区 sample_incomplete_rows.dropna(subset=["total_bedrooms"])
    放弃这个属性sample_incomplete_rows.drop("total_bedrooms", axis=1)
    将缺失的值设置为某个值（0，平均数或者中位数） sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)

数据清洗：
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    复制数据
    使用fit()方法将imputer实例分配到训练集

处理文本和分类属性：
    from sklearn.preprocessing import LabelEncoder(标签),OrdinalEncoder(特征)
    文本转为整数，整数转为矩阵one_hot或者直接用LabelBinarizer

机器学习流程：
    模型选择：培训和评估训练集，选择方差较低的，如果太高则是拟合不足。过低则是过拟合
                使用交叉验证来更好的进行评估
    微调模型：网格搜索（抄参数少） 随机搜索（超参数大） 集成方法

    分析最佳模型及其错误

    通过测试集评估系统

    启动 监控和维护系统

    保存模型 pickle sklearn.externals.joblib

大部分时间用于数据准备，构建监控工具，建立人工评估的流水线以及自动定期训练模型上
'''
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,LabelBinarizer