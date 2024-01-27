#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report
warnings.filterwarnings("ignore")
# In[128]:
df = pd.read_csv("D:\habiba aou\heart2.csv")
df[df.duplicated()]   # We have a duplicated row
df.drop_duplicates(inplace=True)

cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall',"output"]   # Categorical Features

for col in cat_cols:
    # plt.figure(figsize=(5,3))
    sns.countplot(x=col, data=df[cat_cols], hue="output",palette=sns.cubehelix_palette(len(df[col].value_counts())))

con_columns = ["age","trtbps","chol","thalachh","oldpeak","output"]  # Continuing Features

sns.pairplot(df[con_columns], hue="output", palette=sns.color_palette(["#000080","#00ffff"]))


df_continuing = df[con_columns]

scaler = RobustScaler()
df_continuing = scaler.fit_transform(df_continuing.drop(columns="output"))

df_dummy = pd.DataFrame(df_continuing, columns = con_columns[:-1])
df_dummy.head()


df_dummy = pd.concat([df_dummy, df.loc[:, "output"]], axis = 1)
df_dummy.head()


df_melt = pd.melt(df_dummy,id_vars="output",var_name="features",value_name="values")
df_melt.head()


#plt.figure(figsize=(8,6))
sns.swarmplot(x="features",y="values",data=df_melt,hue="output",palette=sns.color_palette(["#2f4f4f","#b22222"]))

#plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), annot=True, fmt=".1f")


cat_cols = ['sex','exng','caa','cp','slp','thall']
con_cols = ["age","thalachh","oldpeak"]

# encoding the categorical columns
df = pd.get_dummies(df, columns = cat_cols, drop_first = True)

# defining the features and target
x = df.drop(columns=['output',"chol","trtbps","fbs",'restecg'])
y = df['output']

# scaling the continuous featuree
x[con_cols] = scaler.fit_transform(x[con_cols])
train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.2, random_state=42)

# Finding the best parameters
logreg0 = LogisticRegression()
grid= {"C": np.logspace(-3,3,7), "penalty":["l1","l2"]}
logreg_cv = GridSearchCV(logreg0,grid,cv=10)
logreg_cv.fit(x,y)


logreg = LogisticRegression(C=logreg_cv.best_params_["C"] , penalty=logreg_cv.best_params_["penalty"])
logreg.fit(train_x,train_y)

# Finding the best parameters
# knn0 = KNeighborsClassifier()
# knn_cv = GridSearchCV(knn0, {"n_neighbors": np.arange(1,50)}, cv=10)
# knn_cv.fit(x,y)

# knn = KNeighborsClassifier(n_neighbors=knn_cv.best_params_["n_neighbors"])
# knn.fit(train_x,train_y)


# Finding the best parameters
grid = {"C":np.arange(1,10,1),'gamma':[0.00001,0.00005, 0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5]}
svm0 = SVC(random_state=42)
svm_cv = GridSearchCV(svm0, grid, cv=10)
svm_cv.fit(x,y)

svm = SVC(C=svm_cv.best_params_["C"], gamma=svm_cv.best_params_["gamma"],random_state=42)
svm.fit(train_x,train_y)


tree = DecisionTreeClassifier()
tree.fit(train_x,train_y)

rf = RandomForestClassifier()
rf.fit(train_x,train_y)

# nb = GaussianNB()
# nb.fit(train_x,train_y)

#logreg_prediction = logreg.predict(test_x)
svm_prediction = svm.predict(test_x)

def make_predict(data):
    data = pd.DataFrame(data)
    new_prediction = svm.predict(data)
    print('ssssssssssssssssssssssssssssssssssssssssssssss' + str(new_prediction))
    return new_prediction


# In[163]:



