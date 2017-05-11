
# coding: utf-8

# In[1]:

cd C:\Users\*******


# Data Processing

# In[2]:

# import relevant packages for processing, modeling and visualization
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sklearn
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, classification_report, roc_auc_score, accuracy_score,
                             f1_score, log_loss, roc_curve, confusion_matrix, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from math import sin, cos, sqrt, atan2, radians
import random
import statsmodels.api as sm
from __future__ import division

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# In[3]:

data=pd.read_csv('NCAA_Tourney_2002-2016.csv')


# In[4]:

#check the data
data.head(5)


# In[5]:

# the location of two places 
def distance(lat1, lon1, lat2, lon2):    

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)    #convert angles x from degree to radians
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    
    return distance


# In[6]:

# calculate the distance between the court to the two teams 
data['dist1'] = data.apply(lambda row: distance(row['host_lat'], row['host_long'], row['team1_lat'], row['team1_long']), axis=1)
data['dist2'] = data.apply(lambda row: distance(row['host_lat'], row['host_long'], row['team2_lat'], row['team2_long']), axis=1)
data['diff_dist'] = data['dist1'] - data['dist2']


# In[7]:

# we wanna to get a more detailed predicted target, thus create score ratio
# ransform two variables to build a new target

data['score_ratio']=data['team1_score']/data['team2_score']


# In[8]:

# variables transformation
data['team1_win_ratio']=data['team1_pt_team_season_wins']/(data['team1_pt_team_season_wins']+data['team1_pt_team_season_losses'])
data['team2_win_ratio']=data['team2_pt_team_season_wins']/(data['team2_pt_team_season_wins']+data['team2_pt_team_season_losses'])

#expectation of winning for each team
data['exp_win1'] = (data['team1_adjoe']**11.5)/ ((data['team1_adjde']**11.5)+(data['team1_adjoe']**11.5))
data['exp_win2'] = (data['team2_adjoe']**11.5)/ ((data['team2_adjde']**11.5)+(data['team2_adjoe']**11.5))


#log 5 , the wining probabilty of team 1 
#P(W) = (A - A B) / (A + B - 2A*B)
data['team1_log5'] = (data['exp_win1'] - (data['exp_win1']*data['exp_win2']))/ (data['exp_win1']+data['exp_win2']-(2*data['exp_win1']*data['exp_win2']))


# In[9]:

# smooth the seed difference, we assume that seed 16 is not 16 times as bad as seed 1.
data['seed_diff'] = data['team1_seed'] - data['team2_seed']

import math
def cube_root(x):
    if x > 0:
        return math.pow(x, float(1)/3)
    elif x < 0:
        return -math.pow(abs(x), float(1)/3)
    else:
        return 0
    
# get the cube root of each seed_diff
data['cuberootseed_diff']=data['seed_diff'].map(cube_root)


# In[10]:

# deal with the rpi rating with null value

data['team2_rpi_rating'].loc[data['team2_rpi_rating']=='--']=np.nan

data['team1_rpi_rating'].loc[data['team1_rpi_rating']=='--']=np.nan
data['team1_rpi_rating']=[float(x) for x in data['team1_rpi_rating']]

data['team2_rpi_rating']=[float(x) for x in data['team2_rpi_rating']]


# In[11]:

# data transformation into format (team1 data / team2 data) ratio 
#we want to emphasize the comparasion between two teams

data['win_percentage_ratio']=data['team1_win_ratio']/data['team2_win_ratio']

data['block_ratio']=data['team1_blockpct']/data['team2_blockpct']

data['fg3_ratio']=data['team1_f3grate']/data['team2_f3grate']

data['steal_ratio']=data['team1_stlrate']/data['team2_stlrate']

data['opp_steal_ratio']=data['team1_oppstlrate']/data['team2_oppstlrate']

data['opp_block_ratio']=data['team1_oppblockpct']/data['team2_oppblockpct']


data['rpi_ratio']=data['team1_rpi_rating']/data['team2_rpi_rating']


# In[12]:

data.head()


# In[14]:

#reindex the data to get the random data

data=data.reindex(np.random.permutation(data.index))


# In[15]:

#get the new index for the data
data=data.reset_index(drop=True)


# In[17]:

# define new name of variables for visualization
def COURT(x):
    if x>0:
        return 'Away'
    elif x<0:
        return 'Home'
    else:
        return 'Equal'
    
data['COURT']=data['diff_dist'].map(COURT)


# In[18]:

#
def RESULT(x):
    if x==1:
        return 'win'
    else :
        return 'loose'

data['RESULT']=data['result'].map(RESULT)  


# In[19]:

def SEED(x):
    if x<0:
        return 'nagative'
    elif x>0:
        return 'positive'
    else:
        return 'Equal seed'

data['SEED']=data['seed_diff'].map(SEED)


# Get a Brief View of Variables

# In[20]:

# visualization of the variables
# imort seaborn for visualization
import seaborn as sns


# In[21]:

# set the background color for visualization
sns.set_style('whitegrid')


# In[22]:

sns.factorplot('COURT',hue='RESULT',data=data,kind='count')
sns.plt.title('Effect of Home or Away')


# In[24]:

sns.factorplot('SEED',hue='RESULT',data=data,kind='count')
sns.plt.title('Effect of cuberoorseed_diff')


# Distribution of Selected Variables

# In[37]:

# distribution of team1_log5
sns.distplot(data['team1_log5'])


# In[38]:

#distribution of seed_diff
sns.distplot(data['cuberootseed_diff'],bins=10)


# In[47]:

sns.distplot(data['win_percentage_ratio'])


# In[40]:

# distribution of block_ratio
sns.distplot(data['block_ratio'])


# In[41]:

# distribution of diff_dist
sns.distplot(data['diff_dist'])


# In[43]:

#distribution of 3_ratio
sns.distplot(data['fg3_ratio'])


# In[44]:

#distribution of steal_ratio
sns.distplot(data['steal_ratio'])


# In[45]:

#distribution of opp_steal_ratio
sns.distplot(data['opp_steal_ratio'])


# In[46]:

# distribution of opp_block_ratio
sns.distplot(data['opp_block_ratio'])


# In[49]:


sns.distplot(data['rpi_ratio'].dropna())


# 
# we choose 'team1-log5', 'cuberootseed_diff', 'win_percentage_ratio', 'block_ratio','diff_dist', '3_ratio','steal_ratio'
# 'opp_block_ratio', 'opp_steal_ratio','rpi_ratio' as the features for model building.
# 
# 

# Standardize the Variables

# In[25]:

scaler=StandardScaler()
data_clean=data[['team1_log5','win_percentage_ratio','diff_dist','block_ratio','cuberootseed_diff','steal_ratio','fg3_ratio','opp_block_ratio','opp_steal_ratio','rpi_ratio','result','score_ratio']]


# In[26]:

data_clean=data_clean.dropna()
data_clean.reset_index(drop=True)


# In[27]:

# X for the explanatory variables and Y for the response variables.
X=scaler.fit_transform(data_clean[['team1_log5','win_percentage_ratio','diff_dist','block_ratio','cuberootseed_diff','steal_ratio','fg3_ratio','opp_block_ratio','opp_steal_ratio','rpi_ratio']])
Y=data_clean['result']


# Select the traning part and testing part

# In[28]:

#spilt the data into training and testing
from sklearn.model_selection import train_test_split


# In[29]:

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=4,test_size=0.2)


# check the distribution of Y

# In[30]:

Y_train.mean()


# In[31]:

Y_test.mean()


# Check the significance of variables in regression model

# In[32]:

from sklearn.feature_selection import f_regression


# In[33]:

f_regression(X,Y)


# The result shows that all variables are significant

# Model Building 

#  model 1 -Logistic Regression

# In[34]:

from sklearn.metrics import classification_report


# In[35]:

logit=LogisticRegression()


# In[36]:

#build the model
logit.fit(X_train,Y_train)


# In[37]:

#performance of Logistic Regression


# In[39]:

pred = logit.predict_proba(X_test)[:,1]
prediction = logit.predict(X_test)


# In[40]:

accuracy_score(Y_test,prediction)


# In[41]:

log_loss(Y_test,pred)


# In[42]:

logit_cross_validation= cross_val_score(logit,X,Y,cv=6,scoring='accuracy')


# In[43]:

#overall performance of logistic regression
logit_cross_validation.mean()


# model 2 - Naive Bayes

# In[44]:

# build the model
from sklearn.naive_bayes import GaussianNB
NB=GaussianNB()
NB.fit(X_train,Y_train)


# In[45]:

prediction=NB.predict(X_test)
pred=NB.predict_proba(X_test)[:,1]


# In[46]:

#performance of Naive Bayes


# In[47]:

accuracy_score(Y_test,prediction)


# In[48]:

log_loss(Y_test,pred)


# In[49]:

NB_cross_validation=cross_val_score(NB,X,Y,cv=6,scoring='accuracy')


# In[50]:

#overall performance of Naive Bayes
NB_cross_validation.mean()


# model 3-Neural Network

# In[51]:

from sklearn.neural_network import MLPClassifier


# In[52]:

#build the model
clf=MLPClassifier()
clf.fit(X_train,Y_train)


# In[53]:

prediction=clf.predict(X_test)
pred=clf.predict_proba(X_test)[:,1]


# In[54]:

accuracy_score(Y_test,prediction)


# In[55]:

log_loss(Y_test,pred)


# In[56]:

clf_cross_validation=cross_val_score(clf,X,Y,cv=6,scoring='accuracy')


# In[57]:

#overall performance of Neural Network
clf_cross_validation.mean()


# model 4-KNN

# In[58]:

from sklearn.neighbors import KNeighborsClassifier
#select the optimized K
k_range=range(1,150)


# In[59]:

k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X,Y,cv=6,scoring='accuracy')
    k_scores.append(scores.mean())


# In[60]:

plt.plot(k_range,k_scores)
plt.xlabel('Value of k for knn')
plt.ylabel('cross validation score')


# In[64]:

#build the model
knn=KNeighborsClassifier(n_neighbors=140)
knn.fit(X_train,Y_train)


# In[65]:

prediction=knn.predict(X_test)
pred=knn.predict_proba(X_test)[:,1]


# In[66]:

log_loss(Y_test,pred)


# In[67]:

knn_cross_validation=cross_val_score(knn,X,Y,cv=6,scoring='accuracy').mean()


# In[68]:

# overall performance of K-Nearest-Neighbors
knn_cross_validation


# model 5-Support Vector Machine

# In[69]:

from sklearn.svm import SVC


# In[70]:

#build the model
svc=SVC(probability=True)
svc.fit(X_train,Y_train)


# In[71]:

prediction=svc.predict(X_test)
pred=svc.predict_proba(X_test)[:,1]


# In[72]:

svc_cross_validation=cross_val_score(svc,X,Y,cv=6,scoring='accuracy')


# In[73]:

svc_cross_validation.mean()


# In[74]:

#overall performance of Support Vector Machine
log_loss(Y_test,pred)


# model 6- Linear Regression using (team1 score / team2 score)

# In[77]:

from sklearn.linear_model import LinearRegression
lreg=LinearRegression()


# In[84]:

#build the model
y=data_clean[['score_ratio','result']]
lreg.fit(X,y['score_ratio'])


# In[85]:

def ratio(x):
    if x>1.:
        return 1
    else:
        return 0


# In[86]:

y['prediction']=lreg.predict(X)


# In[87]:

y['predict_result']=y['prediction'].apply(ratio)


# In[88]:


accuracy_score(y['result'],y['predict_result'])


# In[89]:

recall_score(y['result'],y['predict_result'])


# In[90]:

precision_score(y['result'],y['predict_result'])


# Prediction for the new season

# In[91]:

data_2017=pd.read_csv('NCAA_2017.csv')


# In[92]:

data_2017.head()


# In[93]:

data_2017['dist1'] = data_2017.apply(lambda row: distance(row['host_lat'], row['host_long'], row['team1_lat'], row['team1_long']), axis=1)
data_2017['dist2'] = data_2017.apply(lambda row: distance(row['host_lat'], row['host_long'], row['team2_lat'], row['team2_long']), axis=1)
data_2017['diff_dist'] = data_2017['dist1'] - data_2017['dist2']


# In[94]:

data_2017['diff_dist'].head()


# In[95]:

data_2017['team1_win_ratio']=data_2017['team1_pt_team_season_wins']/(data_2017['team1_pt_team_season_wins']+data_2017['team1_pt_team_season_losses'])
data_2017['team2_win_ratio']=data_2017['team2_pt_team_season_wins']/(data_2017['team2_pt_team_season_wins']+data_2017['team2_pt_team_season_losses'])

#expectation of winning for each team
data_2017['exp_win1'] = (data_2017['team1_adjoe']**11.5)/ ((data_2017['team1_adjde']**11.5)+(data_2017['team1_adjoe']**11.5))
data_2017['exp_win2'] = (data_2017['team2_adjoe']**11.5)/ ((data_2017['team2_adjde']**11.5)+(data_2017['team2_adjoe']**11.5))


#log 5 , the wining probabilty of team 1 
#P(W) = (A - A B) / (A + B - 2A*B)
data_2017['team1_log5'] = (data_2017['exp_win1'] - (data_2017['exp_win1']*data_2017['exp_win2']))/ (data_2017['exp_win1']+data_2017['exp_win2']-(2*data_2017['exp_win1']*data_2017['exp_win2']))


# In[96]:

data_2017['seed_diff'] = data_2017['team1_seed'] - data_2017['team2_seed']


# In[97]:

data_2017['cuberootseed_diff']=data_2017['seed_diff'].map(cube_root)


# In[98]:

data_2017['team1_rpi_rating']=[float(x) for x in data_2017['team1_rpi_rating']]

data_2017['team2_rpi_rating']=[float(x) for x in data_2017['team2_rpi_rating']]


# In[99]:

data_2017['win_percentage_ratio']=data_2017['team1_win_ratio']/data_2017['team2_win_ratio']
data_2017['block_ratio']=data_2017['team1_blockpct']/data_2017['team2_blockpct']
data_2017['fg3_ratio']=data_2017['team1_fg3rate']/data_2017['team2_fg3rate']
data_2017['steal_ratio']=data_2017['team1_stlrate']/data_2017['team2_stlrate']
data_2017['opp_steal_ratio']=data_2017['team1_oppstlrate']/data_2017['team2_oppstlrate']
data_2017['opp_block_ratio']=data_2017['team1_oppblockpct']/data_2017['team2_oppblockpct']


data_2017['rpi_ratio']=data_2017['team1_rpi_rating']/data_2017['team2_rpi_rating']


# In[100]:

data_2017_used=data_2017[['team1_log5','win_percentage_ratio','diff_dist','block_ratio','cuberootseed_diff','steal_ratio','fg3_ratio','opp_block_ratio','opp_steal_ratio','rpi_ratio']]


# In[101]:

X_2017=scaler.fit_transform(data_2017_used)


# Predict the result of season 2017

# In[102]:

logit_2017=LogisticRegression()


# In[103]:

logit_2017.fit(X,Y)


# In[ ]:



