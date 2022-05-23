#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import *


# In[2]:


df=pd.read_csv('Loan Prediction Dataset.csv')


# In[3]:


df.head(20)


# In[4]:


df.shape


# In[5]:


df.skew(axis=0,skipna=True) # negative means left and positive means rightly skewed


# In[6]:


df.kurt(axis=0)


# In[7]:


df.head()


# In[8]:


df.loc[df['Loan_ID'] == 'LP002315']


# In[9]:


duplicate = df['Loan_ID'].duplicated().any()


# In[10]:


print(duplicate)


# In[11]:


unique = df['Loan_ID'].unique()


# In[12]:


df.head()


# In[13]:


df.isnull().sum()


# In[14]:


# there are various methods for dealing with missing values i.e.,
# 1. Deleting the missing values
# 2. Deleting the rows or columns of missing values
# 3. Replacing of Mean, Median or Mode
# 4. Replacing with forward & backward values 
# 5. Missing values can be imputed with using Interpolation, there are methods like Linear, quadratic and polynomial 
# and some more


# In[15]:


#deleting the missing values, rows and columns will lose some useful information, so it is not a good option


# In[16]:


# df = df.dropna(axis=0)(axis=0 for rows and axis = 1 for columns and 
# also we can delete specific column or row like df.dropna(['row'/'column'], axis = 0/1))
# df.isnull().sum()


# In[17]:


#replacing with Mean value but this cannot handle categorical columns and 
#also if there are outliers then first treat outliers then replace with Mean.


# In[18]:


#df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mean()) # use for numerical values only
#df['LoanAmount']=df['LoanAmount'].fillna(df['Credit_History'].mean())
#df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Credit_History'].mean())
#df.isnull().sum()


# In[19]:


#Replacing with Mode is useful for categorical features & fill values with most occuring values


# In[20]:


df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])
df.isnull().sum()


# In[21]:


df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].median()) 
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Credit_History'].median())
df.isnull().sum()


# In[22]:


df.skew(axis=0,skipna=True)


# In[23]:


df.head(20)


# In[24]:


df.describe()


# In[25]:


drop_A=df.index[df["ApplicantIncome"] == 0].tolist()
 
c=drop_A
df=df.drop(df.index[c])
df.head(20)


# In[26]:


df["ApplicantIncome"].fillna(0.0).astype(int)


# In[27]:


df.head()


# In[28]:


#Box plot and Distribution plot can be used to identify appropriate technique for missing values imputation


# In[29]:


import seaborn as sns


# In[30]:


sns.boxplot(df.ApplicantIncome)


# In[31]:


sns.distplot(df.ApplicantIncome)


# In[32]:


# For handling categorical missing values SimpleImputer method can be use(use for replacing most frequent values like Mode)


# In[33]:


from sklearn.impute import SimpleImputer


# In[34]:


imputer = SimpleImputer(strategy='most_frequent')


# In[35]:


imputer.fit(df)


# In[36]:


df.head(20)


# In[37]:


from sklearn.preprocessing import LabelEncoder
cols = ['Loan_ID','Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])
df.head(20)


# In[38]:


df = df.replace(to_replace = "3+" , value = 4)


# In[39]:


df.head(30)


# In[40]:


from sklearn.preprocessing import OrdinalEncoder


# In[41]:


from sklearn.preprocessing import LabelEncoder


# In[42]:


X = df.drop(columns=["Loan_ID",'Loan_Status'], axis=1)
y = df['Loan_Status']


# In[43]:


from sklearn.preprocessing import StandardScaler, normalize


# In[44]:


std_scaler = StandardScaler()
df = std_scaler.fit_transform(df)


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,stratify= y, random_state=1)


# In[47]:


print("Train", X_train.shape, y_train.shape)


# In[48]:


print("Test", X_test.shape, y_test.shape)


# In[ ]:





# In[49]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)


# In[50]:


#from sklearn.tree import DecisionTreeClassifier
#model = DecisionTreeClassifier()


# In[51]:


model.fit(X_train, y_train)


# In[52]:


y_pred = model.predict(X_test)


# In[53]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error


# In[54]:


conMat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(conMat)


# In[55]:


accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
#print("Accuracy: %.2f" % (accuracy*100))


# In[56]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut #cross validation
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


# In[57]:


score = cross_val_score(model, X, y, cv=5)
print("Cross validation is",np.mean(score)*100)


# In[58]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut #cross validation
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


# In[59]:


from sklearn.pipeline import Pipeline


# In[60]:


import sklearn.metrics as metrics


# In[61]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[62]:


from sklearn.metrics import classification_report


# In[63]:


matrix = classification_report(y_test, y_pred )
print("Classification Report: \n", matrix)


# In[64]:


#Dumping the model object


# In[65]:


import pickle
pickle.dump(model, open("model.pkl", "wb"))


# In[66]:


#Reloading the model object


# In[67]:


model = pickle.load(open("model.pkl", "rb"))
print(model)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




