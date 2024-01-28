#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv(r"C:\Users\USER\Downloads\archive (9).zip")
data


# In[3]:


data.info()


# In[4]:


data.dropna(inplace=True)
data


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


dummies=pd.get_dummies(data.ocean_proximity)


# In[8]:


dummies.head()


# In[11]:


final_data=pd.concat([data,dummies],axis=1)


# In[12]:


final_data.head()


# In[13]:


final_data=final_data.drop(['ocean_proximity','ISLAND'],axis=1)


# In[14]:


final_data.head()


# In[15]:


final_data['bedroom_ratio']=final_data['total_bedrooms']/final_data['total_rooms']
final_data['household_rooms']=final_data['total_rooms']/final_data['households']


# In[16]:


final_data.head()


# In[17]:


final_data.corr()


# In[18]:


plt.figure(figsize=(10,10))
sns.heatmap(final_data.corr(),annot=True)


# In[19]:


from sklearn.model_selection import train_test_split


# In[23]:


x=final_data.drop(['median_house_value'],axis=1)
y=final_data['median_house_value']


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[27]:


len(x_train)


# In[29]:


(y_train)


# In[30]:


len(x_test)


# In[31]:


y_test


# In[32]:


from sklearn.linear_model import LinearRegression


# In[33]:


reg=LinearRegression()


# In[34]:


reg.fit(x_train,y_train)


# In[36]:


reg.coef_


# In[37]:


reg.intercept_


# In[39]:


reg.score(x_test,y_test)


# In[ ]:





# In[43]:


reg.predict(x_test)


# In[41]:


y_test


# In[47]:


reg.predict([x_test.iloc[3]])


# In[ ]:




