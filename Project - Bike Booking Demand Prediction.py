#!/usr/bin/env python
# coding: utf-8

# In[41]:


#importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

#To ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[21]:


path='S:/BikeBookingData.csv'
df=pd.read_csv(path, parse_dates=['Date'],encoding="ISO-8859-1")


# In[22]:


df.head()


# In[23]:


df.tail()


# In[24]:


df.shape


# In[25]:


df.columns.tolist()


# In[26]:


df.info()


# In[27]:


df.describe()


# In[28]:


df.duplicated().sum()


# In[29]:


df.isna().sum()


# In[30]:


# Find categorical variables
categorical_variables = [var for var in df.columns if df[var].dtype=='O']
print(categorical_variables)
len(categorical_variables)


# In[31]:


# Checking number of categories in each categorical variables
categorical_variables_df=df[categorical_variables]
for i in categorical_variables_df.columns:
    print(categorical_variables_df[i].value_counts())
    print('--'*50)


# In[34]:


# Finding numerical variables
numerical_variables=[var for var in df.columns if var not in categorical_variables]
print('There are {} numerical variables'.format(len(numerical_variables)))
print('--'*50)
print(numerical_variables)


# In[35]:


# Checking number of categories
numerical_variables_df=df[numerical_variables]
for i in numerical_variables_df.columns:
    print(numerical_variables_df[i].value_counts())
    print('--'*50)


# In[37]:


# Checking for outliers in numerical variables using boxplot
from scipy.stats import norm

# Removing "Date" variable from numerical variable
num_var=[var for var in numerical_variables_df.columns if var not in ["Date"]]

# Plotting Box and Distribution plot 
for var in num_var:
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    ax=sns.boxplot(data=df[var])
    ax.set_title(f'{var}')
    ax.set_ylabel(var)

    plt.subplot(1,2,2)
    ax=sns.distplot(df[var], fit=norm)
    ax.set_title(f'skewness of {var} : {df[var].skew()}')
    ax.set_xlabel(var)
    plt.show()


# The variables Rented Bike Count, Wind speed (m/s), Solar Radiation (MJ/m2), Rainfall (mm) and Snowfall (cm) have outliers. Rainfall (mm) and Snowfall (cm) have a flat interquartile range; it is best to remove those columns.

# In[42]:


# Using Inter Quartile Range for removing outliers from numerical variables

# Defining outlier features
outlier_var=['Booked Bike Count', 'Wind speed (m/s)', 'Solar Radiation (MJ/m2)']

# Capping dataset
for i in outlier_var:
    # Findling IQR
    Q1=df[i].quantile(0.25)
    Q3=df[i].quantile(0.75)
    IQR=Q3-Q1
    
    # Defining upper and lower limit
    lower_limit =df[i].quantile(0.25)-1.5*IQR
    upper_limit =df[i].quantile(0.75)+1.5*IQR
    
    # Applying lower and upper limit to each variables
    df.loc[(df[i] > upper_limit),i] = upper_limit
    df.loc[(df[i] < lower_limit),i] = lower_limit


# In[43]:


# Checking outliers for after removing it

for var in outlier_var:
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    ax=sns.boxplot(data=df[var])
    ax.set_title(f'{var}')
    ax.set_ylabel(var)

    plt.subplot(1,2,2)
    ax=sns.distplot(df[var], fit=norm)
    ax.set_title(f'skewness of {var} : {df[var].skew()}')
    ax.set_xlabel(var)
    plt.show()


# In[ ]:


Outliers are successfully removed from datasets.


# In[44]:


# Final Basic description of dataset
df.describe()


# In[45]:


# Checking unique values from each variable
for i in df.columns:
    print(f'{i} : {df[i].unique()}')
    print('--'*50)


# In[46]:


# Performing feature engineering on feature Date
df['day']=df['Date'].dt.day
df['month']=df['Date'].dt.month
df['year']=df['Date'].dt.year


# In[47]:


# Dropping original  Date feature after performing feature engineering
df.drop(columns='Date', axis=1, inplace=True)


# In[ ]:


The variables day, month, and year were created from the variable Date, and the original variables were deleted. 


# ## <u>EDA<u>

# #### Univariate Analysis

# In[49]:


# Obtaing target variable
excluded_variables=[var for var in df.columns if len(df[var].value_counts()) > 15]
excluded_variables


# In[50]:


target_variables=[var for var in df.columns if var not in excluded_variables]
target_variables


# In[51]:


# Defining r to autofit the number and size of plots
r = int(len(target_variables)/3 +1)


# In[52]:


# Defining a function to Notate the percent count of each value on the bars
def annot_percent(axes):
    '''Takes axes as input and labels the percent count of each bar in a countplot'''
    for p in plot.patches:
        total = sum(p.get_height() for p in plot.patches)/100
        percent = round((p.get_height()/total),2)
        x = p.get_x() + p.get_width()/2
        y = p.get_height()
        plot.annotate(f'{percent}%', (x, y), ha='center', va='bottom')


# In[53]:


# Plotting the countplots for each variable in target_variables
plt.figure(figsize=(18,r*3))
for n,var in enumerate(target_variables):
    plot = plt.subplot(r,3,n+1)
    sns.countplot(x=df[var]).margins(y=0.15)
    plt.title(f'{var.title()}',weight='bold')
    plt.tight_layout()
    annot_percent(plot)


# **Observations :**
# - Customers favour booking motorcycles equally in all seasons.
# - When there are no holidays, customers choose to book motorcycles. Customers hardly ever use the bikes they book while traveling on holiday.
# - Nearly all consumers preferred to book bikes during functional hours.
# - Bicycle booking is popular all month long.
# - Booking bicycles was not very popular in 2017, but it increased by 83.02 percent in 2018.

# #### **<u>Bivariate Analysis<u>**

# In[58]:


# Plotting graph of 'Hour' against 'Rented Bike Count'
plt.figure(figsize=(7,7))
ax=sns.barplot(x="Hour", y="Booked Bike Count",data=df)
ax.set_title('Barplot of Hour against Booked Bike Count')
plt.show()


# **Observations :**
# - At night, customers do not prefer to book bikes.
# - Customers do not prefer booking bikes in the mornings 4 and 5, but from 7, 8, and 9, the use of booking bikes increases, possibly due to working people going to the office, and it is the same in the evenings 5, 6, and 7, because people are travelling from the office to home. Overall, the booked bike was the most frequently used during office in and out times.
# - Customers mostly book bikes for transportation in the evening. 

# In[63]:


# Checking effect of temperature(°C) on rented bike use
plt.figure(figsize=(15,8))
ax=sns.lineplot(x="Temperature(C)", y="Booked Bike Count",data=df)
ax.set_title('Barplot of Temperature(C) against Booked Bike Count')
plt.show()


# **Observations :**
# - Most customers book a bike when the temperature is normal, but when the temperature is below normal, people do not use book a bike. 

# In[64]:


# Checking effect of each variable on use of booking bike
target_variables=[var for var in df.columns if var not in ['Booked Bike Count', 'Seasons', 'Holiday', 'Functioning Day', 'year']]
for var in target_variables:
    plt.figure(figsize=(15,8))
    ax=sns.lineplot(x=df[var], y=df["Booked Bike Count"],data=df)
    ax.set_title(f'lineplot of {var} against Booked Bike Count')
    plt.show()


# **Observations :**
# - Customers who travel most commonly use booking bikes in the morning at 8 a.m. and in the evening at 6 p.m.
# - When the humidity level is between 10% and 18%, people prefer to book bikes.
# - When wind speed is between 2 m/s and 3.5 m/s, people consistently use booking bikes, and it is at its peak when wind speed is normal, which is 3.2 m/s.
# - Booking a bike is the best option for customers in dew point temperatures ranging from 12°C to 18°C. The use of a booking bike increases with increasing dew point temperatures, but it still reaches normal dew point temperatures.
# - According to the graph, solar radiation has no effect on customer use of rented bikes.
# - When it's not raining, people prefer booking bikes the most.
# - When there is no snowfall, most people opt to book bikes. However, the majority of customers prefer to book bikes when it snows up to 4 cm.
# - In the first 10 days of the month, most booking bikes are used by customers. Customers consistently use booking bikes in the last 15 days of the month. 
# - In June, most booking bikes are used through the year, followed by October. Customers' use of booking bikes is at its peak from April to September. 

# In[65]:


# Plotting graph of 'Visibility (10m)' against 'Booked Bike Count'
plt.figure(figsize=(7,7))
ax=sns.scatterplot(x="Visibility (10m)", y="Booked Bike Count",data=df)
ax.set_title('Visibility (10m) against Booked Bike Count')
plt.show()


# **Observation :**
# - The count of booked bikes on that day is unaffected by the day's visibility, but when visibility exceeds 1750, use of booking bikes increases more than usual.

# In[66]:


# Plotting bar plot for variables

# Defining target variables
target=[var for var in df.columns if var in ['Seasons', 'Holiday', 'Functioning Day', 'year']]

for var in target:
    plt.figure(figsize=(7,7))
    ax=sns.barplot(x=df[var], y='Booked Bike Count', data=df)
    ax.set_title(f'{var} v/s Booked Bike Count')
    plt.show()


# **Observations :**
# - During the summer and autumn seasons, most people book bikes. During the winter, fewer people choose to book bikes.
# - Even when there is no holiday other than a holiday, people book bikes. The use of booking bikes on holidays is lower than on non-holiday days.
# - Almost every booking bike is used during its functional hours.
# - The use of booking bikes increased by three times in 2018 compared to 2017.

# ## **<u>Data Pre-processing<u>**

# #### **<i>[1] Checking Distribution of each feature and transform it to normal distribution<i>**

# In[70]:


# Ckecking distribution of numerical feature
# Defining numerical contineous variables
num_variables=['Booked Bike Count', 'Temperature(C)', 'Humidity(%)',
       'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(C)',
       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']

# Plotting distribution plot for each numerical variables
from scipy.stats import norm
for var in num_variables:
    plt.figure(figsize=(5,5))
    ax=sns.distplot(df[var], fit=norm)
    ax.axvline(df[var].mean(), color='magenta', linestyle='dashed', linewidth=2)
    ax.axvline(df[var].median(), color='cyan', linestyle='dashed', linewidth=2)
    ax.set_title(f'skewness of {var} : {df[var].skew()}')
    plt.show()


# In[72]:


# import module
from tabulate import tabulate
 
# assign data
mydata = [
    ["Booked Bike Count", "1.0032657583531357", 'Slightly'],
    ["Temperature(°C", "-0.19832553450003293", 'Nearly Symmetrical'],
    ["Humidity(%)", "0.05957897257708239",'Nearly Symmetrical'],
      ["Wind speed (m/s)", "0.7147003872817881",'Slightly'],
    ['Visibility (10m)','-0.7017864489502947','Slightly'],['Dew point temperature(°C)',' -0.3672984396624286', 'Nearly Symmetrical'],['Solar Radiation (MJ/m2)','1.2673461302699969', 'Slightly'],
    ['Rainfall(mm)','14.5332','Large'], ['Snowfall (cm)','8.440800781484777', 'Large']
]
 
# create header
head = ["Feature Name", "Skew", 'Skew-Type']
 
# display table
print(tabulate(mydata, headers=head, tablefmt="grid"))


# In[74]:


# Transforming distribution to normal using different transformations

# For positively skewed data
df['Booked Bike Count']=(df['Booked Bike Count']+1).transform(np.sqrt)
df['Wind speed (m/s)']=(df['Wind speed (m/s)']+1).transform(np.log)
df['Solar Radiation (MJ/m2)']=(df['Solar Radiation (MJ/m2)']+1).transform(np.log)

# For negatively skewed data
df['Visibility (10m)']=(max(df['Visibility (10m)']+1)-df['Visibility (10m)']).transform(np.sqrt)

# For large skewed data
df['Rainfall(mm)']=(df['Rainfall(mm)']+1).transform(np.log)
df['Snowfall (cm)']=(df['Snowfall (cm)']+1).transform(np.log)


# In[75]:


# Checking distribution after transformed data to normal distribution
from scipy.stats import norm
for var in num_variables:
    plt.figure(figsize=(5,5))
    ax=sns.distplot(df[var], fit=norm)
    ax.axvline(df[var].mean(), color='magenta', linestyle='dashed', linewidth=2)
    ax.axvline(df[var].median(), color='cyan', linestyle='dashed', linewidth=2)
    ax.set_title(f'skewness of {var} : {df[var].skew()}')
    plt.show()


# By applying log transformations, we give variables a distribution that is close to being normal.

# #### **<i>[2] Checking relationship between independent and dependent variables is linear<i>**

# In[76]:


# Plotting regression plot 
for n, var in enumerate([var for var in num_variables if var not in ['Booked Bike Count']]):
    plt.figure(figsize=(20,15))
    plt.subplot(3,3,n+1)
    # Finding correlation of independant variables with dependant variable
    dependant_var=df['Booked Bike Count']
    independant_var=df[var]
    correlation=independant_var.corr(dependant_var)
    ax=sns.regplot(x=df[var], y=df['Booked Bike Count'], data=df, line_kws={"color": "red"})
    ax.set_title(f'{var} v/s Booked Bike Count correlation : {correlation}')
    plt.show()


# All numerical variables are correlated with the dependent variable Rented Bike Count, but Solar Radiation (MJ/m2), Dew Point Temperature (°C) and Temperature (°C) are highly correlated with the dependent variable, which is good for a linear machine learning model.

# #### **<i>[3] Checking multicollinearity in independant variables<i>**

# ##### **<u>Correlation Heatmap<u>**

# In[77]:


# Plotting a correlation heatmap for the dataset
plt.figure(figsize=(15,8))
correlation=df.corr()
sns.heatmap(correlation, vmin=-1, cmap='coolwarm', annot=True)
plt.show()


# **Observations :**
# - Dew point temperature (°C) and temperature (°C) have a strong correlation. A moderate correlation exists between humidity (%) and dew point temperature (°C). The variables year and Dew point temperature (°C) have a weak correlation.

# ##### **<u>Variance inflation factor(VIF)<u>**

# In[78]:


# Variance inflation factor(VIF) to detects multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)


# In[79]:


# Calculating Variance inflation factor for dataset
calc_vif(df[[i for i in df.describe().columns if i not in ['Booked Bike Count']]])


# - Variable year have a high variance inflation factor, so we can remove it.

# In[80]:


# Dropping year feature to reduce Variance inflation factor
calc_vif(df[[i for i in df.describe().columns if i not in ['Rented Bike Count','year']]])


# - Variable Temperature(°C), Humidity(%) and Dew point temperature(°C) have a high variance inflation factor, so we can remove Dew point temperature(°C) variable to reduce VIF.

# In[82]:


# Dropping 'Dew point temperature(°C)' feature to reduce Variance inflation factor
calc_vif(df[[i for i in df.describe().columns if i not in ['Booked Bike Count','year','Dew point temperature(°C)']]])


# - From above, variables year,Dew point temperature(°C), and Humidity(%) have a high variance inflation factor, so we can remove them.

# In[84]:


# Drop high VIF variables from dataset
df.drop(columns=['year','Dew point temperature(C)', 'Humidity(%)'], axis=1,inplace=True)


# In[85]:


# Checking dataset
df.head()


# #### **<i>[4] Features encoding<i>**

# In[86]:


# Addressing categorical variables from the dataset
categorical_variables=df.describe(include=['object']).columns
categorical_variables


# In[87]:


# Checking categories in each categorical features
for var in categorical_variables:
    print(df[var].value_counts())
    print('--'*50)


# In[88]:


# Plotting count plot for each categories in categorical_variables
for var in categorical_variables:
    plt.figure(figsize=(7,5))
    plot=plt.subplot(111)
    ax=sns.countplot(x=df[var])
    annot_percent(plot)
    plt.show()


# - Because categorical variables have a limited number of categories, label encoding can be used instead of one-hot encoding. One hot encoding is used when there are quite large categories in categorical variables.

# In[89]:


# Encoding categorical_variables using label encoding
# Mapping the categorical variables
df['Seasons'] = df['Seasons'].map({'Spring':1,'Summer':2,'Autumn':3,'Winter':4})
df['Holiday'] = df['Holiday'].map({'No Holiday':0,'Holiday':1})
df['Functioning Day'] = df['Functioning Day'].map({'Yes':1,'No':0})


# In[92]:


# Final Dataset
df.head()


# - This is the final dataset we will use to build a machine learning model.

# #### **<i>[5] Data Splitting<i>**

# In[95]:


# Separating independant variables and dependant variable

# Creating the dataset with all dependent variables
dependent_variable = 'Booked Bike Count'

# Creating the dataset with all independent variables
independent_variables = list(set(df.columns.tolist()) - {dependent_variable})

# Create the data of independent variables
X = df[independent_variables].values
# Create the data of dependent variable
y = df[dependent_variable].values


# In[96]:


# Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)


# In[97]:


# Checking shape of split
print(f'Shape of X_train : {X_train.shape}')
print(f'Shape of X_test : {X_test.shape}')
print(f'Shape of y_train : {y_train.shape}')
print(f'Shape of y_test : {y_test.shape}')


# - We divided the dataset into 20% for model testing and 80% for training.

# In[98]:


# Checking values of splitted dataset
X_train[0:5]


# In[99]:


# Checking values of splitted dataset
X_test[0:5]


# #### **<i>[6] Data Scaling<i>**

# In[100]:


#Transforming data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# - MinMaxScaler preserves the shape of the original distribution. It doesn't meaningfully change the information embedded in the original data. So we used MinMaxScaler for scaling the dataset.

# In[101]:


# Checking values of splitted dataset after normalisation
X_train[0:5]


# In[102]:


# Checking values of splitted dataset after normalisation
X_test[0:5]


# ## **<u>ML Model Implementation<u>**

# In[104]:


# Definig function for evaluating model

# Importing necessary librery
from sklearn import metrics
#Evaluate Metrics
def evaluate_model(true, predicted):  
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    mae = metrics.mean_absolute_error(true, predicted)
    r2_square = metrics.r2_score(true, predicted)
    print('Mean squared error (MSE):', mse)
    print('Root mean squared error (RMSE):', rmse)
    print('Mean absolute error (MAE):', mae)
    print('R2 Square', r2_square)


# ### **[1] Linear Regression**

# #### [1.1] train_test_split

# In[105]:


# Fitting Regression model to training set
from sklearn.linear_model import LinearRegression
lin_reg_tts = LinearRegression()
lin_reg_tts.fit(X_train, y_train)


# In[106]:


# Score of the model
lin_reg_tts_score=lin_reg_tts.score(X_train, y_train)
print(f'Score of the model : {lin_reg_tts_score}')


# In[107]:


# Predicting results for train and test set
y_train_pred_tts=lin_reg_tts.predict(X_train)
y_test_pred_tts=lin_reg_tts.predict(X_test)


# In[113]:


# Evaluating model
print(f'Training set evaluation result\n____________________________________')
evaluate_model(y_train, y_train_pred_tts)
print(f'\nTest set evaluation result\n____________________________________')
evaluate_model(y_test, y_test_pred_tts)


# In[114]:


# Evaluating Adjusted R2
from sklearn.metrics import r2_score

print(f'Train set evaluation result\n____________________________________')
print("Adjusted R2 : ",1-(1-r2_score((y_train), (y_train_pred_tts)))*((X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1)))
print('____________________________________')
print(f'Test set evaluation result\n____________________________________')
print("Adjusted R2 : ",1-(1-r2_score((y_test), (y_test_pred_tts)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)))


# In[115]:


# Finding mean of Residuals
print(f'Mean of Residuals for test set\n____________________________________')
residuals = y_test-y_test_pred_tts
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))


# **Note :-** Mean of Residuals is very very close to zero, so all good here.

# In[116]:


# Plotting graph of Actual (true) values and predicted values
plt.figure(figsize=(15,5))
plt.plot(y_test[0:100])
plt.plot(y_test_pred_tts[0:100])
plt.title('Actual (true) values v/s predicted values')
plt.legend(["Actual", "Predicted"])
plt.show()


# In[117]:


# Plotting graph of Actual (true) values and predicted values
plt.figure(figsize=(15,5))
sns.scatterplot(x=y_test, y=y_test_pred_tts)
plt.title('Relationship between Actual (true) values and predicted values')
plt.show()


# - We get the conclusion that the linear model fails to perform well based on the above model score and graph.
# - Model provided only 62% of the performance score, and the mean of the residual was not close to zero, indicating that the model was not well trained to make predictions. 

# ### **[2] Decision Tree**

# In[119]:


# Fitting Decision tree Regression model to training set
from sklearn.tree import DecisionTreeRegressor
dtree_tts=DecisionTreeRegressor(criterion='squared_error',max_leaf_nodes=10, random_state=0)
dtree_tts.fit(X_train, y_train)


# In[120]:


# Score of the model
dtree_tts_score=dtree_tts.score(X_train, y_train)
print(f'Score of the model : {dtree_tts_score}')


# In[121]:


# Predicting results for train and test set
y_train_pred_tts=dtree_tts.predict(X_train)
y_test_pred_tts=dtree_tts.predict(X_test)


# In[122]:


# Evaluating model
print(f'Training set evaluation result\n____________________________________')
evaluate_model(y_train, y_train_pred_tts)
print(f'Test set evaluation result\n____________________________________')
evaluate_model(y_test, y_test_pred_tts)


# In[123]:


# Evaluating Adjusted R2
from sklearn.metrics import r2_score

print(f'Train set evaluation result\n____________________________________')
print("Adjusted R2 : ",1-(1-r2_score((y_train), (y_train_pred_tts)))*((X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1)))
print('____________________________________')
print(f'Test set evaluation result\n____________________________________')
print("Adjusted R2 : ",1-(1-r2_score((y_test), (y_test_pred_tts)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)))


# In[124]:


# Finding mean of Residuals
print(f'Mean of Residuals for test set\n____________________________________')
residuals = y_test-y_test_pred_tts
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))


# In[125]:


# Plotting graph of Actual (true) values and predicted values
plt.figure(figsize=(15,5))
plt.plot(y_test[0:100])
plt.plot(y_test_pred_tts[0:100])
plt.title('Actual (true) values v/s predicted values')
plt.legend(["Actual", "Predicted"])
plt.show()


# #### Cross Validation

# In[127]:


# Fitting Decision tree regression model to training set
from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(random_state=0)
param_dict = {'max_depth':[4,6,8],
              'min_samples_split':[50,100,150],
              'min_samples_leaf':[40,50]}
dtree_reg = GridSearchCV(dtree, param_dict, verbose=1, scoring='neg_mean_squared_error', cv=5)
dtree_reg.fit(X_train, y_train)


# In[128]:


# Best estimators
dtree_reg_best_est=dtree_reg.best_estimator_
print(f'The best estimator values : {dtree_reg_best_est}')


# In[129]:


# best fit values
dtree_reg_best_params=dtree_reg.best_params_
print(f'The best fit values: {dtree_reg_best_params}')


# In[130]:


# Mean cross-validated score of the best_estimator of model
dtree_reg_score=dtree_reg.best_score_
print(f" The negative mean squared error is: {dtree_reg_score}")


# In[131]:


# Predicting results for train and test set
y_train_pred=dtree_reg.predict(X_train)
y_test_pred=dtree_reg.predict(X_test)


# In[132]:


# Evaluating model
print(f'Training set evaluation result\n____________________________________')
evaluate_model(y_train, y_train_pred)
print(f'Test set evaluation result\n____________________________________')
evaluate_model(y_test, y_test_pred)


# In[133]:


# Evaluating Adjusted R2
from sklearn.metrics import r2_score

print(f'Train set evaluation result\n____________________________________')
print("Adjusted R2 : ",1-(1-r2_score((y_train), (y_train_pred)))*((X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1)))
print('____________________________________')
print(f'Test set evaluation result\n____________________________________')
print("Adjusted R2 : ",1-(1-r2_score((y_test), (y_test_pred)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)))


# In[134]:


# Finding mean of Residuals
print(f'Mean of Residuals for test set\n____________________________________')
residuals = y_test-y_test_pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))


# In[135]:


# Plotting graph of Actual (true) values and predicted values
plt.figure(figsize=(15,5))
plt.plot(y_test[0:100])
plt.plot(y_test_pred[0:100])
plt.title('Actual (true) values v/s predicted values')
plt.legend(["Actual", "Predicted"])
plt.show()


# - A model decision tree using GridSearchCV performs well based on the above model score and graph.

# In[ ]:




