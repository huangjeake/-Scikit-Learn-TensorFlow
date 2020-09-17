from sklearn.datasets import load_iris,fetch_20newsgroups
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def max_min():
    """
    归一化
    :return:
    """
    data = pd.read_csv('./data/dating.txt')
    print(data)
    # print(data)
    #1.实例化
    transfer = MinMaxScaler(feature_range=(2,3))
    print(transfer)
    #2.进行转化调用fit_transform
    ret_data = transfer.fit_transform(data[['milage','Liters','Consumtime']])
    print("最小值最大值归一化处理的结果：\n", ret_data)

def standard():
    """
    标准化
    :return:
    """
    data = pd.read_csv('./data/dating.txt')
    # print(data)
    #1.实例化
    transfer = StandardScaler()
    #2.进行转化调用fit_transform
    ret_data = transfer.fit_transform(data[['milage','Liters','Consumtime']])
    plt.show()
    print("最小值最大值归一化处理的结果：\n", ret_data)
    print("标准化方差：\n", transfer.var_)
    print("标准化平均值：\n", transfer.mean_)


if __name__ == '__main__':
    # max_min()
    standard()
