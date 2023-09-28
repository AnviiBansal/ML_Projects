#!/usr/bin/env python
# coding: utf-8

# In[4]:


import math, time, random, datetime

# data analysis and wrangling
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

#To ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[5]:


# visualization
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

#import for interactive plotting
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler


# In[7]:


# machine learning
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.metrics import confusion_matrix,classification_report


# # Import and Inspect Data

# In[9]:


df = pd.read_csv("S:/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[10]:


df.head()


# In[11]:


df.shape


# # Exploratory Data Analysis

# In[9]:


ProfileReport(df)


# - Job level is strongly correlated with total working hours
# - Monthly income is strongly correlated with Job level
# - Monthly income is strongly correlated with total working hours
# - Age is stongly correlated with monthly income

# In[10]:


# drop the unnecessary columns
df.drop(['EmployeeNumber','Over18','StandardHours','EmployeeCount'],axis=1,inplace=True)


# In[11]:


df['Attrition'] = df['Attrition'].apply(lambda x:1 if x == "Yes" else 0 )
df['OverTime'] = df['OverTime'].apply(lambda x:1 if x =="Yes" else 0 )


# In[12]:


attrition = df[df['Attrition'] == 1]
no_attrition = df[df['Attrition']==0]


# ### Visualization of Categorical Features 

# In[13]:


def categorical_column_viz(col_name):
    
    f,ax = plt.subplots(1,2, figsize=(10,6))
  
    # Count Plot
    df[col_name].value_counts().plot.bar(cmap='Set2',ax=ax[0])
    ax[1].set_title(f'Number of Employee by {col_name}')
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel(f'{col_name}')
    
    # Attrition Count per factors
    sns.countplot(col_name, hue='Attrition',data=df, ax=ax[1], palette='Set2')
    ax[1].set_title(f'Attrition by {col_name}')
    ax[1].set_xlabel(f'{col_name}')
    ax[1].set_ylabel('Count')


# In[14]:


categorical_column_viz('BusinessTravel')


# In[15]:


categorical_column_viz('Department')


# In[16]:


categorical_column_viz('EducationField')


# In[17]:


categorical_column_viz('Education')


# In[18]:


categorical_column_viz('EnvironmentSatisfaction')


# In[19]:


categorical_column_viz('Gender')


# In[20]:


categorical_column_viz('JobRole')


# In[21]:


categorical_column_viz('JobInvolvement')


# In[22]:


categorical_column_viz('MaritalStatus')


# In[23]:


categorical_column_viz('NumCompaniesWorked')


# In[24]:


categorical_column_viz('OverTime')


# In[25]:


categorical_column_viz('StockOptionLevel')


# In[26]:


categorical_column_viz('TrainingTimesLastYear')


# In[27]:


categorical_column_viz('YearsWithCurrManager')


# ### Visualization of Numerical Features 

# In[28]:


def numerical_column_viz(col_name):
    f,ax = plt.subplots(1,2, figsize=(18,6))
    sns.kdeplot(attrition[col_name], label='Employee who left',ax=ax[0], shade=True, color='palegreen')
    sns.kdeplot(no_attrition[col_name], label='Employee who stayed', ax=ax[0], shade=True, color='salmon')
    
    sns.boxplot(y=col_name, x='Attrition',data=df, palette='Set3', ax=ax[1])


# In[29]:


numerical_column_viz("Age")


# In[30]:


numerical_column_viz("Age")


# In[31]:


numerical_column_viz("DailyRate")


# In[32]:


numerical_column_viz("DistanceFromHome")


# In[33]:


numerical_column_viz("MonthlyIncome")


# In[34]:


numerical_column_viz("HourlyRate")


# In[35]:


numerical_column_viz("JobInvolvement")


# In[36]:


numerical_column_viz("PercentSalaryHike")


# In[37]:


numerical_column_viz("Age")


# In[38]:


numerical_column_viz("DailyRate")


# In[39]:


numerical_column_viz("TotalWorkingYears")


# In[40]:


numerical_column_viz("YearsAtCompany")


# In[41]:


numerical_column_viz("YearsInCurrentRole")


# In[42]:


numerical_column_viz("YearsSinceLastPromotion")


# In[43]:


numerical_column_viz("YearsWithCurrManager")


# ### Visualization of Categorical vs Numericals Features 

# In[44]:


def categorical_numerical(numerical_col, categorical_col1, categorical_col2):
    

    f,ax = plt.subplots(1,2, figsize=(20,8))
    
    g1= sns.swarmplot( categorical_col1, numerical_col,hue='Attrition', data=df, dodge=True, ax=ax[0], palette='Set2')
    ax[0].set_title(f'{numerical_col} vs {categorical_col1} separeted by Attrition')
    g1.set_xticklabels(g1.get_xticklabels(), rotation=90) 

    
    g2 = sns.swarmplot( categorical_col2, numerical_col,hue='Attrition', data=df, dodge=True, ax=ax[1], palette='Set2')
    ax[1].set_title(f'{numerical_col} vs {categorical_col1} separeted by Attrition')
    g2.set_xticklabels(g2.get_xticklabels(), rotation=90) 


# In[45]:


categorical_numerical('Age','Gender','MaritalStatus')


# In[46]:


categorical_numerical('Age','JobRole','EducationField')


# In[47]:


categorical_numerical('MonthlyIncome','Gender','MaritalStatus')


# ## Feature Engineering

# In[48]:


# 'EnviornmentSatisfaction', 'JobInvolvement', 'JobSatisfacction', 'RelationshipSatisfaction', 'WorklifeBalance' can be clubbed into a single feature 'TotalSatisfaction'

df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] + 
                            df['JobInvolvement'] + 
                            df['JobSatisfaction'] + 
                            df['RelationshipSatisfaction'] +
                            df['WorkLifeBalance']) /5 

# Drop Columns
df.drop(['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'], axis=1, inplace=True)


# In[49]:


categorical_column_viz('Total_Satisfaction')


# In[50]:


df.Total_Satisfaction.describe()


# In[51]:


# Convert Total satisfaction into boolean
# median = 2.8
# x = 1 if x >= 2.8

df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply(lambda x:1 if x>=2.8 else 0 ) 
df.drop('Total_Satisfaction', axis=1, inplace=True)


# In[52]:


# It can be observed that the rate of attrition of employees below age of 35 is high

df['Age_bool'] = df['Age'].apply(lambda x:1 if x<35 else 0)
df.drop('Age', axis=1, inplace=True)


# In[53]:


# It can be observed that the employees are more likey the drop the job if dailtRate less than 800

df['DailyRate_bool'] = df['DailyRate'].apply(lambda x:1 if x<800 else 0)
df.drop('DailyRate', axis=1, inplace=True)


# In[54]:


# Employees working at R&D Department have higher attrition rate

df['Department_bool'] = df['Department'].apply(lambda x:1 if x=='Research & Development' else 0)
df.drop('Department', axis=1, inplace=True)


# In[55]:


# Rate of attrition of employees is high if DistanceFromHome > 10

df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x:1 if x>10 else 0)
df.drop('DistanceFromHome', axis=1, inplace=True)


# In[56]:


# Employees are more likey to drop the job if the employee is working as Laboratory Technician

df['JobRole_bool'] = df['JobRole'].apply(lambda x:1 if x=='Laboratory Technician' else 0)
df.drop('JobRole', axis=1, inplace=True)


# In[57]:


# Employees are more likey to the drop the job if the employee's hourly rate < 65

df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x:1 if x<65 else 0)
df.drop('HourlyRate', axis=1, inplace=True)


# In[58]:


# Employees are more likey to the drop the job if the employee's MonthlyIncome < 4000

df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply(lambda x:1 if x<4000 else 0)
df.drop('MonthlyIncome', axis=1, inplace=True)


# In[59]:


# Rate of attrition of employees is high if NumCompaniesWorked < 3

df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply(lambda x:1 if x>3 else 0)
df.drop('NumCompaniesWorked', axis=1, inplace=True)


# In[60]:


# Employees are more likey to the drop the job if the employee's TotalWorkingYears < 8

df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply(lambda x:1 if x<8 else 0)
df.drop('TotalWorkingYears', axis=1, inplace=True)


# In[61]:


# Employees are more likey to the drop the job if the employee's YearsAtCompany < 3

df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply(lambda x:1 if x<3 else 0)
df.drop('YearsAtCompany', axis=1, inplace=True)


# In[62]:


# Employees are more likey to the drop the job if the employee's YearsInCurrentRole < 3

df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply(lambda x:1 if x<3 else 0)
df.drop('YearsInCurrentRole', axis=1, inplace=True)


# In[63]:


# Employees are more likey to the drop the job if the employee's YearsSinceLastPromotion < 1

df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply(lambda x:1 if x<1 else 0)
df.drop('YearsSinceLastPromotion', axis=1, inplace=True)


# In[64]:


# Employees are more likey to the drop the job if the employee's YearsWithCurrManager < 1

df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply(lambda x:1 if x<1 else 0)
df.drop('YearsWithCurrManager', axis=1, inplace=True)


# In[65]:


df['Gender'] = df['Gender'].apply(lambda x:1 if x=='Female' else 0)


# In[66]:


df.drop('MonthlyRate', axis=1, inplace=True)
df.drop('PercentSalaryHike', axis=1, inplace=True)


# In[67]:


convert_category = ['BusinessTravel','Education','EducationField','MaritalStatus','StockOptionLevel','OverTime','Gender','TrainingTimesLastYear']
for col in convert_category:
        df[col] = df[col].astype('category')


# In[68]:


df.info()


# In[69]:


#separate the categorical and numerical data
X_categorical = df.select_dtypes(include=['category'])
X_numerical = df.select_dtypes(include=['int64'])
X_numerical.drop('Attrition', axis=1, inplace=True)


# In[70]:


y = df['Attrition']


# In[71]:


# One HOt Encoding Categorical Features

onehotencoder = OneHotEncoder()

X_categorical = onehotencoder.fit_transform(X_categorical).toarray()
X_categorical = pd.DataFrame(X_categorical)
X_categorical


# In[72]:


#concat the categorical and numerical values

X_all = pd.concat([X_categorical, X_numerical], axis=1)
X_all.head()


# In[73]:


X_all.info()


# ### Split Data

# In[74]:


X_train,X_test, y_train, y_test = train_test_split(X_all,y, test_size=0.30)


# In[75]:


print(f"Train data shape: {X_train.shape}, Test Data Shape {X_test.shape}")


# In[76]:


X_train.head()


# ## Train Data

# In[77]:


# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo(algo, X_train,y_train, cv):
    
    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    
    # Cross Validation 
    train_pred = model_selection.cross_val_predict(algo,X_train,y_train,cv=cv,n_jobs = -1)
    
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    
    return train_pred, acc, acc_cv


# ### Logistic Regression

# In[78]:


# Logistic Regression
start_time = time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), X_train,y_train, 10)
log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))


# # Predict Data using Logistic Regression

# In[95]:


model = LogisticRegression().fit(X_train, y_train)


# In[96]:


predictions = model.predict(X_test)


# In[97]:


pred_df = pd.DataFrame(index=X_test.index)


# In[98]:


pred_df['Attrition'] = predictions
pred_df.head()


# In[99]:


# Cross-validation accuracy metric
score = round(metrics.accuracy_score(y_test, predictions) * 100, 2)


# In[100]:


print("Accuracy: %s" % score)


# In[101]:


print(classification_report(y_test, predictions))


# In[102]:


# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

