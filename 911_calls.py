#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('whitegrid')

plt.rcParams['figure.figsize'] = (6, 4)


# In[2]:


data= pd.read_csv('C:\\Users\\91829\\Downloads\\911.csv')
data.head()


# In[3]:


data.info()


# Basic Analysis
# 
# Let’s check out the top 5 zipcodes for calls.

# In[5]:


data['zip'].value_counts().head()


# The top townships for the calls were as follows:

# In[7]:


data['twp'].value_counts().head()


# For 650k + entries, how many unique call titles did we have?

# In[8]:


data['title'].nunique()


# # Data Wrangling for Feature Creation

# In[10]:


data['reason'] = data['title'].apply(lambda x: x.split(':')[0])
data.tail()


# Now, let’s find out the most common reason for 911 calls, according to our dataset.

# In[11]:


data['reason'].value_counts().head()


# In[12]:


sns.countplot(data['reason'])


# In[17]:


type(data['timeStamp'][0])


# As the timestamps are still string types, it’ll make our life easier if we convert it to a python DateTime object, so we can extract the year, month, and day information more intuitively.

# In[18]:


data['timeStamp'] = pd.to_datetime(data['timeStamp'])


# For a single DateTime object, we can extract information as follows.

# In[20]:


time = data['timeStamp'].iloc[0]

print('Hour:',time.hour)
print('Month:',time.month)
print('Day of Week:',time.dayofweek)


# Now let’s create new features for the above pieces of information.

# In[27]:


data['Hour'] = data['timeStamp'].apply(lambda x: x.hour)
data['Month'] = data['timeStamp'].apply(lambda x: x.month)
data['Day of Week'] = data['timeStamp'].apply(lambda x: x.dayofweek)

data.head()


# In[28]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
data['Day of Week'] = data['Day of Week'].map(dmap)

data.head()


# Let’s combine the newly created features, to check out the most common call reasons based on the day of the week.

# In[29]:


sns.countplot(data['Day of Week'],hue=data['reason'])

plt.legend(bbox_to_anchor=(1.25,1))


# In[30]:


sns.countplot(data['Month'],hue=data['reason'])

plt.legend(bbox_to_anchor=(1.25,1))


# In[32]:


byMonth = data.groupby(by='Month').count()


# In[33]:


byMonth['e'].plot.line(y='e')
plt.title('Calls per Month')
plt.ylabel('Number of Calls')


# In[34]:


byMonth.reset_index(inplace=True)

sns.lmplot(x='Month',y='e',data=byMonth)
plt.ylabel('Number of Calls')


# So, it does seem that there are more emergency calls during the holiday seasons.

# In[35]:


data['Date']=data['timeStamp'].apply(lambda x: x.date())
data.head(2)


# In[36]:


data.groupby(by='Date').count()['e'].plot.line(y='e')
plt.legend().remove()
plt.tight_layout()


# Let’s create a heatmap for the counts of calls on each hour, during a given day of the week.

# In[49]:


day_hour = data.pivot_table(values='lat',index='Day of Week',columns='Hour',aggfunc='count')

day_hour


# In[50]:


sns.heatmap(day_hour)

plt.tight_layout()


# We see that most calls take place around the end of office hours on weekdays. We can create a clustermap to pair up similar Hours and Days.

# In[51]:


sns.clustermap(day_hour)


# In[ ]:




