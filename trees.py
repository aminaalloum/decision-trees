#!/usr/bin/env python
# coding: utf-8

# In[158]:


import pandas as pd
df=pd.read_csv("./Downloads/titanic-passengers.csv", sep=";")
df


# In[159]:


df.isnull().sum()


# In[160]:


df["Embarked"].value_counts()


# In[161]:


df["Embarked"].fillna("S",inplace=True)


# In[162]:


mean=round(df["Age"].mean())
mean


# In[163]:


df["Age"].fillna(round(mean),inplace=True)


# In[164]:


df["Cabin"].value_counts()


# In[165]:


df["Cabin"].fillna("G6",inplace=True)
df["Cabin"]


# In[166]:


df


# In[167]:


df["Sex"]=df["Sex"].map({"male":1,"female":0}).values.reshape(-1,1)
df["Survived"]=df["Survived"].map({"Yes":0,"No":1})
df


# In[168]:


from sklearn.model_selection import train_test_split
y=df[["Survived"]]
x=df[["Sex","Age","Fare"]]


# In[169]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[170]:


df.isnull().sum()


# In[171]:


from sklearn import tree
model=tree.DecisionTreeClassifier()


# In[172]:


from sklearn import tree 
clf=tree.DecisionTreeClassifier()
clf=clf.fit(x_train,y_train)


# In[173]:


import graphviz
dotfile = open("dtree.pdf", 'w')
print(x.columns)
ts=tree.export_graphviz(clf, out_file = dotfile, feature_names=x.columns, filled=True, 
                    rounded=True, impurity=False, class_names=["Survived","no survived"])
graph=graphviz.Source(dotfile)
graph.render("dtree.pdf",view=True)


# In[174]:


y_pred=clf.predict(x_test)
from sklearn import metrics 
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[175]:


df["Parch"].value_counts()


# In[177]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
data=le.fit_transform(df["Cabin"])
print(data)


# In[178]:


le.classes_


# In[179]:


df=df.drop("Cabin",axis=1)


# In[180]:


df["Cabin"]=data
df


# In[181]:


clf=clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Acrruacy:",metrics.accuracy_score(y_test,y_pred))


# In[182]:


#0.765363<0.770949 the seconde accruacy is higher than the first but the result it's can
#still changing after reexucate the code


# In[186]:


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df["Embarked"].values.reshape(1,-1)
ohe.fit_transform(df[["Embarked"]])


# In[187]:


ohe.fit_transform(df[["Cabin"]])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[189]:


from sklearn.ensemble import RandomForestRegressor
x=["Embarked","Age","Survived"]
y=['sex']
regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(x,y)


# In[ ]:





# In[ ]:




