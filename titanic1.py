
# %%
# 读取数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#测试数据
train = pd.read_csv('..//data//homework//train.csv')
train.head()
#测试数据集
test = pd.read_csv("../data/homework/test.csv")

#data = pd.read_csv('..//data//homework//train.csv')
df = train.copy()
df.sample(10)
# %%
# 去除无用特征
train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
train.info()
# %%
# 替换/删除空值，这里是删除
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
train.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# 把categorical数据通过one-hot变成数值型数据
# 很简单，比如sex=[male, female]，变成两个特征,sex_male和sex_female，用0, 1表示


print ('训练数据集:',train.shape,'测试数据集:',test.shape)
rowNum_train=train.shape[0]
rowNum_test=test.shape[0]
print('kaggle训练数据集有多少行数据：',rowNum_train,
     ',kaggle测试数据集有多少行数据：',rowNum_test,)
#合并数据集，方便同时对两个数据集进行清洗
full = train.append( test , ignore_index = True )#使用append进行纵向堆叠

print(full.isnull().sum())
#年龄(Age)
full['Age']=full['Age'].fillna( full['Age'].mean() )
#船票价格(Fare)
full['Fare'] = full['Fare'].fillna( full['Fare'].mean() )

full['Embarked'] = full['Embarked'].fillna( 'S' )
full['Cabin'] = full['Cabin'].fillna( 'U' )
sex_mapDict={'male':1,
            'female':0}
#map函数：对Series每个数据应用自定义的函数计算
full['Sex']=full['Sex'].map(sex_mapDict)
embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies( full['Embarked'] , prefix='Embarked' )
full = pd.concat([full,embarkedDf],axis=1)
full.drop('Embarked',axis=1,inplace=True)
pclassDf = pd.DataFrame()

#使用get_dummies进行one-hot编码，列名前缀是Pclass
pclassDf = pd.get_dummies( full['Pclass'] , prefix='Pclass' )
full = pd.concat([full,pclassDf],axis=1)

#删掉客舱等级（Pclass）这一列
full.drop('Pclass',axis=1,inplace=True)


full_X = pd.concat( [
                     full['Age'],
                     pclassDf,#客舱等级
                     full['Fare'],#船票价格
                     full['Sex'],#性别
                     embarkedDf,#登船港口
                    ] , axis=1 )
sourceRow=891
#原始数据集：特征
source_X = full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']

#预测数据集：特征
pred_X = full_X.loc[sourceRow:,:]

# 分离训练集与测试集
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(source_X, source_y, test_size=0.2, random_state=1)
print('X_train: {}'.format(np.shape(X_train)))
print('y_train: {}'.format(np.shape(y_train)))
print('X_test: {}'.format(np.shape(X_test)))
print('y_test: {}'.format(np.shape(y_test)))
# %%
# 建立模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = dict()

models['SVM'] = SVC(kernel='rbf')  # SVM这里我们搞个最常用的
models['KNeighbor'] = KNeighborsClassifier(n_neighbors=5)  # n_neighbors表示neighbor个数
models['RandomForest'] = RandomForestClassifier(n_estimators=100)  # n_estimators表示树的个数
# %%
# 定义函数以评估模型
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np


def plot_cm(model, y_true, y_pred, name=None):
    """画混淆矩阵
    :param model: 分类模型
    :param y_true: 标签实际值
    :param y_pred: 标签预测值
    :param name: 模型名称
    """
    _, ax = plt.subplots()
    if name is not None:
        ax.set_title(name)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.show()
    return None


def plot_cm_ratio(model, y_true, y_pred, name=None):
    """画混淆矩阵（按占各类型比例）
    :param model: 分类模型
    :param y_true: 标签实际值
    :param y_pred: 标签预测值
    :param name: 模型名称
    """
    _, ax = plt.subplots()
    if name is not None:
        ax.set_title(name)
    cm = confusion_matrix(y_true, y_pred)
    cm_ratio = np.zeros(cm.shape)
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            cm_ratio[i, j] = cm[i, j] / cm[i].sum()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ratio, display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.show()
    return None


def model_perf(model, y_true, y_pred, name=None):
    """返回模型分类准确率，tpr，fpr
    """
    if name is not None:
        print('For model {}: \n'.format(name))
    cm = confusion_matrix(y_true, y_pred)
    for i in range(len(model.classes_)):
        # TODO: Add comments
        tp = cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tn = cm.sum() - tp - fp - fn
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        acc = (tp + tn) / cm.sum()
        print('For class {}: \n TPR is {}; \n FPR is {}; \n ACC is {}. \n'
        .format(model.classes_[i], tpr, fpr, acc))
    return None


def ovo_eval(model, name=None):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    plot_cm(model, y_test, prediction, name)
    plot_cm_ratio(model, y_test, prediction, name)
    model_perf(model, y_test, prediction, name)
    print('Overall Accuracy: {}'.format(model.score(X_test, y_test)))
# %%
# 评估各模型性能
for name, model in models.items():
    ovo_eval(model, name)
