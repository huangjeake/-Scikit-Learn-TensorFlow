'''
pandas去重数据:
    object.drop_duplicates(subset=[‘method’,‘year’],keep=‘first’,inplace=True)
    subset参数是一个列表，这个列表是需要你填进行相同数据判断的条件。就比如我选的条件是method和year，即 method值和year的值相同就可被判定
    为一样的数据。keep的取值有三个 分别是 first、last、false；keep=first时，保留相同数据的第一条。keep=last时，保存相同数据的最后一条。
    keep=false时，所有相同的数据都不保留。inplace=True时，会对原数据进行修改。否则，只返回视图，不对原数据修改

pandas操作：
    data.dropna(axis=1) 根据删除空列
    x= ['时间','A-C OH_39（电芯头部)','A-C OH_40（电芯头部)','A-C OH_40（电芯尾部)']
    data.drop(labels=x, axis=1, inplace=True)删除制定列
    data.set_index('条码',inplace=True)设置索引 并且删除旧的索引




'''

# 每列数据与索引画图

import pandas as pd
import plotly
import plotly.graph_objs as go

import numpy as np
data_origin = pd.read_csv(r'C:\Users\ext_renqq\Desktop\完整数据\CCD2020-10-05.CSV')
# Create a trace
for k in data_origin.columns.values:
#     print(k)
#     a = data[k]
    trace = go.Scatter(
        x = data_origin[k].index,
        y = data_origin[k].values,
        mode = 'lines'
    #      mode = 'markers'
    )

    data = [trace]

    plotly.offline.plot(data,filename = "./project_photo/" + k + ".html")