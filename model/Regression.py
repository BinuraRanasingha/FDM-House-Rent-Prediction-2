#!/usr/bin/env python
# coding: utf-8

# In[71]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Materials/')
import pandas as pd
df = pd.read_csv("new_house.csv") 


# In[64]:


df.head(5)


# In[16]:


import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# In[53]:


df.shape


# In[54]:


df.dtypes


# In[72]:


df['Posted On']=pd.to_datetime(df['Posted On'])


# In[73]:


correlation = df.corr(method='pearson')
columns = correlation.nlargest(10, 'Rent').index
correlation_map = np.corrcoef(df[columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True,
                      fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)

plt.show()


# In[74]:


df.describe()


# In[75]:


#Since Rent and Size columns has large values
df['Rent'] = np.log(df['Rent'])
df['Size'] = np.log(df['Size'])


# In[76]:


df.describe()


# In[77]:


columns=columns.drop(['size_scale','rent_scale'])
columns=columns.drop(['Rent'])


# In[78]:


columns


# In[79]:


X = df[columns]
Y = df['Rent'].values


# In[83]:


#Splitting the datset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=42)


# # Importing the Regression Libraries

# In[84]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


# Linear Regression

# In[98]:


from sklearn.metrics import mean_squared_error
linear = LinearRegression()
linear.fit(X_train, Y_train)
pred=linear.predict(X_test)
#model is good if R square is high, and MSE is lowest
linear_rsq = (linear.score(X_test,Y_test))*100

linear_mse = mean_squared_error(pred, Y_test)


# Lasso Regression

# In[99]:


lasso = Lasso()
lasso.fit(X_train, Y_train)
lasso_pred=lasso.predict(X_test)
#model is good if R square is high, and MSE is lowest
lasso_rsq = (lasso.score(X_test,Y_test))*100

lasso_mse = mean_squared_error(lasso_pred, Y_test)


# ElasticNet Regression

# In[100]:


el = ElasticNet()
el.fit(X_train, Y_train)
el_pred=el.predict(X_test)
#model is good if R square is high, and MSE is lowest
el_rsq = (el.score(X_test,Y_test))*100

el_mse = mean_squared_error(el_pred, Y_test)


# DecisionTreeRegressor

# In[101]:


dec = DecisionTreeRegressor()
dec.fit(X_train, Y_train)
dec_pred=dec.predict(X_test)
#model is good if R square is high, and MSE is lowest
dec_rsq = (dec.score(X_test,Y_test))*100

dec_mse = mean_squared_error(dec_pred, Y_test)


# KNeighborsRegressor

# In[102]:


kn = KNeighborsRegressor()
kn.fit(X_train, Y_train)
kn_pred=kn.predict(X_test)
#model is good if R square is high, and MSE is lowest
kn_rsq = (kn.score(X_test,Y_test))*100

kn_mse = mean_squared_error(kn_pred, Y_test)


# GradientBoostingRegressor

# In[103]:


grad = GradientBoostingRegressor()
grad.fit(X_train, Y_train)
grad_pred=grad.predict(X_test)
#model is good if R square is high, and MSE is lowest
grad_rsq = (grad.score(X_test,Y_test))*100

grad_mse = mean_squared_error(grad_pred, Y_test)


# In[107]:



list_reg=['Linear','Lasso','ElasticNet','Decision Tree','KNeighbors','Gradient Boost']
data = {'Regressor': list_reg, 'R Score': [linear_rsq, lasso_rsq, el_rsq,dec_rsq,kn_rsq,grad_rsq], 'MSE':[linear_mse, lasso_mse, el_mse,dec_mse,kn_mse,grad_mse]}
reg_df=pd.DataFrame(data)
reg_df


# In[109]:


import pickle
pickle.dump(grad,open('regression.pkl','wb'))

