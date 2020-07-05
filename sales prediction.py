
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns


# In[2]:
#importing training data set

train=pd.read_csv(r'C:\Resources\Bigmart-Sales-Prediction-master\dataset\Train.csv')


# In[3]:
#returns first five rows of data set

train.head()


# In[4]:
#returns the shape of training data set

train.shape


# In[5]:
#importing test data set

test=pd.read_csv(r'C:\Resources\Bigmart-Sales-Prediction-master\dataset\Test.csv')


# In[6]:


test.head()


# In[7]:


test.shape


# In[8]:

#returns total number of null values in each column of train data set

train.isnull().sum()


# In[9]:

#returns total number of null values in each column of train data set


test.isnull().sum()


# In[10]:


train.describe()


# In[11]:


# plotting correlation matrix

correl=train.corr()
ax=plt.subplots(figsize=(15,9))
sns.heatmap(correl,vmax=0.8,square=True)


# In[12]:


train.head()


# In[13]:

#returns the total number of each item in Item_Fat_Content column

train.Item_Fat_Content.value_counts()  #need to optimize


# In[14]:

#returns the total number of each item in Item_Type column


train.Item_Type.value_counts()


# In[15]:

#returns the total number of each item in Outlet_Identifier column


train.Outlet_Identifier.value_counts()


# In[16]:

#returns the total number of each item in Outlet_Size column

train.Outlet_Size.value_counts()


# In[17]:


train.Outlet_Location_Type.value_counts()


# In[18]:

#returns the total number of each item in Outlet_Type column


train.Outlet_Type.value_counts()


# In[19]:


train.head()


# In[20]:


train.isnull().sum()


# In[21]:

#plots histogram

train.Item_Weight.hist(bins=50)


# In[22]:

#plots histogram

train.Outlet_Size.hist(bins=50)


# In[23]:

#plots histogram

train.Outlet_Size.value_counts()


# In[24]:



Item_Sales=train.Item_Outlet_Sales


# In[25]:

#appending test dataset to train dataset 

data=train.append(test)


# In[26]:

#returns the shape of dataset

data.shape


# In[27]:

#returns the total number of null values in each column of dataset

data.isnull().sum()


# In[28]:


data.isnull().sum()


# In[29]:

#plotting correlation matrix

correlation=data.corr()
sns.heatmap(correlation,vmax=.8,square=True)


# In[30]:

#returns total number of each item in every column of dataset

data.apply(lambda x:len(x.unique()))


# In[31]:

#returns the datatype of each column

data.dtypes


# In[32]:

#returns the label of each column of dataset

data.dtypes.index


# In[33]:

#filtering out categorical columns

categorical_columns=[x for x in data.dtypes.index if data.dtypes[x]=='object']
categorical_columns


# In[34]:

#filtering out categorical columns

categorical_columns=[x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier']]
categorical_columns


# In[35]:


#print frequencies of these categories

for col in categorical_columns:
    print('frequency of categories for variable')
    print(data[col].value_counts())


# In[36]:

#fills the empty values of Item_Weight clolumn with the respective mean of the column
    
data.Item_Weight.fillna(data.Item_Weight.mean(),inplace=True)


# In[37]:


#import mode function
from scipy.stats import mode

#determining the mode of each 

#mapping the items of Outlet_Size column
data.Outlet_Size=data.Outlet_Size.map({'Small':0,'Medium':1,'High':2})

#creating a table with iems of "Outlet_Type" as columns and 'Outlet_Size' as row and mode of each
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]))

#returns "True" if the value of 'Outlet_Size' column is null
miss_bool = data['Outlet_Size'].isnull()

#filling empty values of each column with mode of respective type 
data.loc[miss_bool,'Outlet_Size'] = data.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])


# In[38]:

#returns the total number of empty cells in each column
data.isnull().sum()


# In[39]:

#print items of every column whose count is <30

for i in data.dtypes.index:
    if len(data[i].value_counts())<30:
        print(i,"\n",data[i].value_counts())


# In[40]:

#creating a table with 'Outlet_Type' as column and 'Item_Outlet_Sales' as row
data.pivot_table(index='Outlet_Type',values='Item_Outlet_Sales')


# In[41]:

#plotting histogram 

data.Item_Visibility.hist(bins=50)


# In[42]:

#finding the mean of 'Item_Visibility' column
data.Item_Visibility.mean()


# In[43]:

#finding the location of 0 values of selected column and filling such values with mean of respected caolumn
data.loc[data['Item_Visibility']==0,'Item_Visibility']=data.Item_Visibility.mean()


# In[44]:


data.Item_Type.value_counts()


# In[45]:

#creating new column using Item_Type_Combined by extracting frst two letters of "Item_Identifier"

data['Item_Type_Combined']=data.Item_Identifier.apply(lambda x:x[0:2])

#verifying count of each item_type_combined

data['Item_Type_Combined'].value_counts()


# In[46]:

#updating Item_Type_Combined

data['Item_Type_Combined']=data.Item_Type_Combined.map({'FD':'Food and Drinks','NC':'Non-Consumable','DR':'Drinks'})


# In[47]:


data['Item_Type_Combined'].value_counts()


# In[48]:

#creating new coulmn with how many years it passed since 2013
data['Outlet_Years']=2013-data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# In[49]:


data.Item_Fat_Content.value_counts()


# In[50]:

#updating "Item_Fat_Content" column
data.Item_Fat_Content=data.Item_Fat_Content.replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
data.Item_Fat_Content.value_counts()


# In[51]:


data.loc[data['Item_Type_Combined']=='Non-Consumable','Item_Fat_Content']='Non-Edible'


# In[52]:


data.Item_Fat_Content.value_counts()


# In[53]:

data.head()



# In[54]:


#import library 
#now import labelEncoding
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
data['Outlet']=lb.fit_transform(data['Outlet_Identifier'])
var=['Item_Fat_Content','Outlet_Location_Type','Outlet_Type','Outlet_Size','Item_Type_Combined']
lb=LabelEncoder()
for item in var:
    data[item]=lb.fit_transform(data[item])


# In[55]:


data.head()


# In[56]:

#dropping the values
data.drop(['Outlet_Establishment_Year','Item_Type'],inplace=True,axis=1)


# In[57]:


data.head()
    

# In[58]:


Item_Sales=data.Item_Outlet_Sales


# In[59]:

#selects first 8523 values of dataset as training dataset
train=data.iloc[:8523,:]
train.head()


# In[60]:

#selects second 8523 values of dataset as test dataset
test=data.iloc[8523:,:]


# In[61]:

#dropping selecting column
test.drop('Item_Outlet_Sales',inplace=True,axis=1)


# In[62]:


test.head()


# In[63]:


# A generalization function to prediction and file on sharing
target='Item_Outlet_Sales'
IDcol=['Item_Identifier','Outlet_Identifier']
from sklearn import model_selection ,metrics
def modelfit(alg,dtrain,dtest,predictor,target,IDcol,filename):
    alg.fit(dtrain[predictor],dtrain[target])
    prediction=alg.predict(dtrain[predictor])
    #now cross_validation
    cv_score=model_selection.cross_val_score(alg,dtrain[predictor],dtrain[target],cv=20,scoring='neg_mean_squared_error')
    cv_score=np.sqrt(np.abs(cv_score))
    print(np.sqrt(metrics.mean_squared_error(dtrain[target].values,prediction)))
    print("CV_SCORE : mean - %.4g | std - %.4g | max - %.4g | min - %.4g" % (np.mean(cv_score),np.std(cv_score),np.max(cv_score),np.min(cv_score)))
    dtest[target]=alg.predict(dtest[predictor])
    
    #now export on submission file 
    IDcol.append(target)
    submission=pd.DataFrame({x:dtest[x] for x in IDcol})
    submission.to_csv()
    #submission.to_csv


# In[64]:


#Linear Regression on training set
from sklearn.linear_model import LinearRegression , Ridge,Lasso
predictor=[x for x in train.columns if x not in [target]+IDcol]
alg1=LinearRegression()
modelfit(alg1,train,test,predictor,target,IDcol,'alg1.csv')


# In[65]:


predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')


# In[66]:


from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')


# In[67]:


predictors = ['Item_MRP','Outlet_Type','Outlet','Outlet_Years']
alg4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')
coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')


# In[68]:
