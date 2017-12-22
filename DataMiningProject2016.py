
# coding: utf-8

# #  Data Mining Project
# 
# ### Ntoulos Panagiotis
# 
# #### 
# 
# 
# ## problem approach
# 
#From the dataset which was given, we decided to 'work' with the flight dates. Because between certain periods such as christmas or weekends many people travel.
# 
# 
# ### functions
# 
# 
# 
# 
# 

# In[7]:


import numpy as np
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import codecs
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from datetime import date
from dateutil.easter import *




'''
easter(date,type)
type:
EASTER_JULIAN   = 1
EASTER_ORTHODOX = 2
EASTER_WESTERN  = 3
'''
"""*********************************"""

def monthsbeforeEaster(dataset):
    #calculates the weeks before easter for every year thanks to easter library.
    _weeksbefore=[]
    for i in range(0,dataset.shape[0]):
        dateTokens=dataset[i,0].split('-')
        onedate=date(int(dateTokens[0]),int(dateTokens[1]),int(dateTokens[2]))
        _easter=easter(int(dateTokens[0]),3)
        _subdays=_easter-onedate
        _subdays=abs(round(_subdays.days/30.41,0))
        _weeksbefore.append(_subdays)
    return _weeksbefore
"""***************************************************"""


# In[8]:


def weeksbeforeEaster(dataset):
    #calculates the weeks before easter for every year thanks to easter library.
    _weeksbefore=[]
    for i in range(0,dataset.shape[0]):
        dateTokens=dataset[i,0].split('-')
        onedate=date(int(dateTokens[0]),int(dateTokens[1]),int(dateTokens[2]))
        _easter=easter(int(dateTokens[0]),3)
        _subdays=_easter-onedate
        _subdays=round(_subdays.days/7.0,0)
        _weeksbefore.append(_subdays)
    return _weeksbefore


# In[9]:


def weekday(dataset):
    #Calculates the day of the week[0,1,2,3,4,5,6] for [mon,tue.....sunday]
    _weekday=[]
    for i in range(0,dataset.shape[0]):
         dateTokens=dataset[i,0].split('-')
         onedate=date(int(dateTokens[0]),int(dateTokens[1]),int(dateTokens[2]))
         _weekday.append(onedate.weekday())
    
    return _weekday


# In[10]:


def weeksbeforeChristmas(dataset):
    
    
    #calculates the weeks before christmas
        #date(year,month,day)
        #array.shape[0/1]--> 0--> rows , 1--> columns
    
    
    christmasdate= date(2016,12,25)
    daysbefore=[]
    for i in range(0,dataset.shape[0]):
        dateTokens=dataset[i,0].split('-')
        #We want to compute the days before/after christmas. So we set a constant year.
        onedate=date(2016,int(dateTokens[1]),int(dateTokens[2]))
        subDays=(christmasdate-onedate)
        subDays=round(subDays.days/7.0,0)
        daysbefore.append(subDays)
    return daysbefore


# In[11]:


def monthsbeforeChristmas(dataset):
    
    
    #calculates the weeks before christmas
        #date(year,month,day)
        #array.shape[0/1]--> 0--> rows , 1--> columns
    
    
    christmasdate= date(2016,12,25)
    daysbefore=[]
    for i in range(0,dataset.shape[0]):
        dateTokens=dataset[i,0].split('-')
        #We want to compute the days before/after christmas. So we set a constant year.
        onedate=date(2016,int(dateTokens[1]),int(dateTokens[2]))
        subDays=(christmasdate-onedate)
        subDays=abs(round(subDays.days/30.41,0))
        daysbefore.append(subDays)
    return daysbefore


# In[12]:


def weeksbeforeNewYearsEve(dataset):
    
    
    #calculates the weeks before Newyears Eve
        #date(year,month,day)
        #array.shape[0/1]--> 0--> rows , 1--> columns
    
    
    newyeareve= date(2016,12,31)
    daysbefore=[]
    for i in range(0,dataset.shape[0]):
        dateTokens=dataset[i,0].split('-')
        #We want to compute the days before/after christmas. So we set a constant year.
        onedate=date(2016,int(dateTokens[1]),int(dateTokens[2]))
        subDays=(newyeareve-onedate)
        subDays=round(subDays.days/7.0,0)
        daysbefore.append(subDays)
    return daysbefore


# In[13]:


def monthsbeforeNewYearsEve(dataset):
    
    
    #calculates the weeks before Newyears Eve
        #date(year,month,day)
        #array.shape[0/1]--> 0--> rows , 1--> columns
    
    
    newyeareve= date(2016,12,31)
    daysbefore=[]
    for i in range(0,dataset.shape[0]):
        dateTokens=dataset[i,0].split('-')
        #We want to compute the days before/after christmas. So we set a constant year.
        onedate=date(2016,int(dateTokens[1]),int(dateTokens[2]))
        subDays=(newyeareve-onedate)
        subDays=abs(round(subDays.days/30.41,0))
        daysbefore.append(subDays)
    return daysbefore


# In[14]:


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    We cant use euclidean distance.
    Earth is not flat.
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km 


# ### main()
# 
# 

# In[15]:


df_train =pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

y_train = df_train.as_matrix(['PAX'])#ari8mos epivatwn#
df_train.drop(df_train.columns[[11]], axis=1, inplace=True)



date_train=df_train.as_matrix(['DateOfDeparture'])#input for the functions

date_test=df_test.as_matrix(['DateOfDeparture'])#input for the functions



le = LabelEncoder()

"""*******TRAIN Transform**************"""

le.fit(df_train['Departure'])
df_train['Departure'] = le.transform(df_train['Departure'])
df_train['Arrival'] = le.transform(df_train['Arrival'])

le.fit(df_train['CityDeparture'])
df_train['CityDeparture'] = le.transform(df_train['CityDeparture'])

le.fit(df_train['CityArrival'])
df_train['CityArrival'] =le.transform(df_train['CityArrival'])


Strings_train=df_train.as_matrix(['Departure','Arrival','CityDeparture','CityArrival'])

df_train.drop(df_train.columns[[0,1,2,5,6]], axis=1, inplace=True)

Numbers_train=df_train.as_matrix()

"""**************Test Transform*****************"""

le.fit(df_test['Departure'])
df_test['Departure'] = le.transform(df_test['Departure'])
df_test['Arrival'] = le.transform(df_test['Arrival'])

le.fit(df_test['CityDeparture'])
df_test['CityDeparture'] = le.transform(df_test['CityDeparture'])

le.fit(df_test['CityArrival'])
df_test['CityArrival'] =le.transform(df_test['CityArrival'])


Strings_test=df_test.as_matrix(['Departure','Arrival','CityDeparture','CityArrival'])

df_test.drop(df_test.columns[[0,1,2,5,6]], axis=1, inplace=True)

Numbers_test=df_test.as_matrix()

"""*************************************************"""


# #### new features

# In[16]:


"""***************** MONTHS ***************"""


TR_m_easter = np.asarray(monthsbeforeEaster(date_train))
TS_m_easter = np.asarray(monthsbeforeEaster(date_test))

TR_m_newyear = np.asarray(monthsbeforeNewYearsEve(date_train))
TS_m_newyear = np.asarray(monthsbeforeNewYearsEve(date_test))

TR_m_christmas = np.asarray(monthsbeforeChristmas(date_train))
TS_m_christmas = np.asarray(monthsbeforeChristmas(date_test))

"""*************** END     ******************"""


"""******************** WEEKS ***************"""
TR_easter= np.asarray(weeksbeforeEaster(date_train))
TS_easter= np.asarray(weeksbeforeEaster(date_test))

TR_weekday= np.asarray(weekday(date_train))
TS_weekday= np.asarray(weekday(date_test))

TR_newyear= np.asarray(weeksbeforeNewYearsEve(date_train))
TS_newyear=np.asarray(weeksbeforeNewYearsEve(date_test))

TR_christmas = np.asarray(weeksbeforeChristmas(date_train))
TS_christmas = np.asarray(weeksbeforeChristmas(date_test))
"""****************************** """




#gia to Train set
distance=[]
for i in range(0,8899):
    distance.append(haversine(Numbers_train[i,0], Numbers_train[i,1], Numbers_train[i,2], Numbers_train[i,3]))

t_distance = np.asarray(distance)

#Gia to testSet
te_distance=[]
for i in range(0,2229):
    te_distance.append(haversine(Numbers_test[i,0], Numbers_test[i,1], Numbers_test[i,2], Numbers_test[i,3]))

te_distance = np.asarray(te_distance)



# #### tables for prediction

# In[17]:


#TRAIN
Strings_train= np.concatenate((Strings_train, TR_weekday[:,None]), axis=1)
Strings_train= np.concatenate((Strings_train, TR_m_easter[:,None]), axis=1)
Strings_train= np.concatenate((Strings_train, TR_m_newyear[:,None]), axis=1)
Strings_train= np.concatenate((Strings_train, TR_m_christmas[:,None]), axis=1)

#TEST
Strings_test= np.concatenate((Strings_test, TS_weekday[:,None]), axis=1)
Strings_test= np.concatenate((Strings_test, TS_m_easter[:,None]), axis=1)
Strings_test= np.concatenate((Strings_test, TS_m_newyear[:,None]), axis=1)
Strings_test= np.concatenate((Strings_test, TS_m_christmas[:,None]), axis=1)

enc = OneHotEncoder(sparse=False)
enc.fit(Strings_train)
Strings_train=enc.transform(Strings_train)

Strings_test=enc.transform(Strings_test)


X_train = np.concatenate((Strings_train,TR_easter[:,None]), axis=1)



X_train = np.concatenate((X_train, t_distance[:,None]), axis=1)



X_train = np.concatenate((X_train, TR_christmas[:,None]), axis=1)





"""Telikos X_Train"""

X_train = np.concatenate((X_train, TR_newyear[:,None]), axis=1)



"""X_test"""



X_test = np.concatenate((Strings_test, TS_easter[:,None]), axis=1)



X_test = np.concatenate((X_test, te_distance[:,None]), axis=1)



X_test = np.concatenate((X_test, TS_christmas[:,None]), axis=1)




X_test = np.concatenate((X_test, TS_newyear[:,None]), axis=1)


# #### PREDICTION'S CODE

# In[18]:


from sklearn.ensemble import RandomForestClassifier
randomf=RandomForestClassifier(n_estimators=115,criterion='entropy',class_weight='balanced')
randomf.fit(X_train,y_train.ravel())
y_pred=randomf.predict(X_test)

with codecs.open('y_pred.txt', 'w', encoding='utf-8') as f:
    for i in range(y_pred.shape[0]):
        f.write(unicode(y_pred[i])+'\n')


# best score : 0.5419

