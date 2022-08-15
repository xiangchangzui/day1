# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# 读取数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("..//data//homework//train.csv")
test = pd.read_csv("..//data//homework//test.csv")
#data = pd.read_csv('..//data//homework//train.csv')
#df = data.copy()
#df.sample(10)
# %%
# 去除无用特征
train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare', 'Age', 'Embarked'], inplace=True)
train.info()
# %%
# 替换/删除空值，这里是删除
print('Is there any NaN in the dataset: {}'.format(train.isnull().values.any()))
train.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(train.isnull().values.any()))
print ('训练数据集:',train.shape,'测试数据集:',test.shape)
rowNum_train=train.shape[0]
rowNum_test=test.shape[0]
print('kaggle训练数据集有多少行数据：',rowNum_train,
     ',kaggle测试数据集有多少行数据：',rowNum_test,)
full = train.append( test , ignore_index = True )
print(full.isnull().sum())



full['Age']=full['Age'].fillna( full['Age'].mean() )
pclassDf = pd.get_dummies( full['Pclass'] , prefix='Pclass' )
full = pd.concat([full,pclassDf],axis=1)
sex_mapDict={'male':1,
            'female':0}
#map函数：对Series每个数据应用自定义的函数计算
full['Sex']=full['Sex'].map(sex_mapDict)
pclassDf = pd.DataFrame()

#使用get_dummies进行one-hot编码，列名前缀是Pclass
pclassDf = pd.get_dummies( full['Pclass'] , prefix='Pclass' )
full = pd.concat([full,pclassDf],axis=1)
#删掉客舱等级（Pclass）这一列
full.drop('Pclass',axis=1,inplace=True)


familyDf = pd.DataFrame()
familyDf[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
familyDf[ 'Family_Single' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
familyDf[ 'Family_Small' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
familyDf[ 'Family_Large' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
full = pd.concat([full,familyDf],axis=1)
full.drop('FamilySize',axis=1,inplace=True)
corrDf = full.corr()
corrDf['Survived'].sort_values(ascending =False)
full_X = pd.concat( [pclassDf,#客舱等级
                     familyDf,#家庭大小
                     full['Sex'],#性别
                    ] , axis=1 )
sourceRow=891
#原始数据集：特征
source_X = full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']

#预测数据集：特征
pred_X = full_X.loc[sourceRow:,:]

#建立模型用的训练数据集和测试数据集

size=np.arange(0.6,1,0.1)
scorelist=[[],[],[],[],[],[]]
from sklearn.model_selection import train_test_split
for i in range(0,4):
    train_X, test_X, train_y, test_y = train_test_split(source_X ,
                                                        source_y,
                                                      train_size=size[i],
                                                        random_state=5)
    #逻辑回归
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit( train_X , train_y )
    scorelist[0].append(model.score(test_X , test_y ))
    # 随机森林Random Forests Model
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_X, train_y)
    scorelist[1].append(model.score(test_X, test_y))



plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
color_list = ('red', 'blue')
for i in range(0, 2):
    plt.plot(size, scorelist[i], color=color_list[i])
plt.legend(['逻辑回归', '随机森林'])

plt.xlabel('训练集占比')
plt.ylabel('准确率')
plt.title('不同的模型随着训练集占比变化曲线')
plt.show()
# %%
# 把categorical数据通过one-hot变成数值型数据
# 很简单，比如sex=[male, female]，变成两个特征,sex_male和sex_female，用0, 1表示
#df = pd.get_dummies(df)
#df

# %%
# 相关系数矩阵
# %%
# 相关系数矩阵可视化
# %%
# train-test split
# %%
# build model
# %%
# predict and evaluate