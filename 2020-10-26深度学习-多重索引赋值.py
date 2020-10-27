'''
pandas开发文档
See the `user guide
<https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`

arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
index = = pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))#设置多重索引
df = pd.DataFrame(np.random.randn(3, 8), index=['A', 'B', 'C'], columns=index)#第一个参数是数组，第二个是索引，第三个是
多重索引列名
first        bar                 baz                 foo                 qux
second       one       two       one       two       one       two       one       two
A       0.895717  0.805244 -1.206412  2.565646  1.431256  1.340309 -1.170299 -0.226169
B       0.410835  0.813850  0.132003 -0.827317 -0.076467 -1.187678  1.130127 -1.436737
C      -1.413681  1.607920  1.024180  0.569605  0.875906 -2.211372  0.974466 -2.006747
df = df.T#进行转置
df.loc[('bar','one'),'A'] = 100#第一个参数小括号里面的行索引即条件，第二个是赋值的列名




'''