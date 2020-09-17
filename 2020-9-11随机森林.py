from sklearn.tree import DecisionTreeClassifier
import matplotlib
'''
随机森林的思想是：训练多个弱分类器，通过投票机制来形成一个强分类器，预测结果
当预测器尽可能互相独立时，集成方法的效果最优。获得多种分类器的方法之一就是使用不同的算法进行训练。这会增加它们犯不同类型错误的机会，
从而提升集成的准确率。


np.cumsum累加 np.cumsum(a,axis=0) #按照行累加，行求和
 array([[1, 2, 3],       array([[1, 2, 3], 
        [4, 5, 6]])             [5, 7, 9]])
        
np里面reshape 在/前面运算

plt.plot(attr)是将数组里面的元素分别画出图片

多个弱分类器采用投票的方式可以形成一个强分类器：
    方式一：通过不同算法生成分类器调用VotingClassifier
    from  sklearn.ensemble import RandomForestClassifier,VotingClassifier
    硬投票：根据投票数量将特征分类；VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],voting='hard')
    软投票：根据权重进行投票；需要将svc的probability= True,voting设置为soft
    方式二：通过同一算法对不同训练集训练调用BaggingClassifier生成不同的分类器，需要设置bootstrap=True
    bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, n_jobs=4, random_state=42, bootstrap=True)
    
外包评估：
    随机森林进行采样的时候有些样本无法采集到，所以无需进行交叉验证。创建BaggingClassifier时，设置
    oob_score=True，可以提高模型的准确率。

Random Pathches和随机子空间：
    对于训练实例和特征同时抽样，被成为Random Patches方法 保留所有训练实例，
    但是对特征进行抽样，被称为随机子空间法
    
随机森林是对决策树的集成，构建BaggingClassifier然后在里面调用DecisionTreeClassifier
    .还有一种方法就是直接调用from sklearn.ensemble import  RandomForestClassifier
    RandomForestClassifier里面包含BaggingClassifirer和DecisionTreeClassifier所有的
    超参数。前者用来控制集成，后者用来控制数的生长。
    
    极端随机树：
        对于每个特征使用随机阀值，不再搜索最佳阀值，这种极端的决策树组成的森林被称为极端随机树集成
        通常来说，很难预先知道一个RandomForestClassifier是否会比一个ExtraTreesClassifier更好或是更差。唯一的方法是两种都尝试
        一遍，然后使用交叉验证（还需要使用网格搜索调整超参数）进行比较。

        

'''
from  sklearn.ensemble import RandomForestClassifier,VotingClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import  RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier# 极端随机树集成
from sklearn.model_selection import GridSearchCV# 网格搜索

svm_clf = SVC(gamma='auto', random_state=42, probability=True)