#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# In[41]:


data = pd.read_csv('Downloads\\Mini Project 1\\Final Projects Machine Learning\\Project 15.0\\titanic_train.csv')


# In[42]:


data


# In[13]:


data.describe()


# In[17]:


data.shape


# In[16]:


data.isnull().sum()


# In[18]:


data.info()


# In[19]:


data.head()


# In[33]:


## Exploratory Data Analysis


# In[35]:


plt.figure(figsize=(20,15),facecolor='yellow')
graph=1
for column in data.select_dtypes(include=['int64', 'float64']):
    if graph <=9:  ## limited to the 9 graphs
        ax=plt.subplot(3,3,graph)
        sns.distplot(data[column])
        plt.xlabel(column)
    graph +=1
plt.tight_layout()


# In[26]:


##Outliers
plt.figure(figsize=(20,15),facecolor='green')
columns_to_check = ['Age', 'Fare', 'SibSp', 'Parch']
for i, column in enumerate(columns_to_check, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=data, x=column)
    plt.title(f'Boxplot of {column}', fontsize=15)
    plt.xlabel(column, fontsize=12)

plt.tight_layout()
plt.show()


# In[47]:


##finding the correlation using heatmaps
corr_matrix=data.corr(numeric_only=True)
corr_matrix


# In[50]:


plt.figure(figsize=(10,5))
sns.heatmap(corr_matrix, annot=True, cmap='magma', fmt='.2f')
plt.title('Correlation')
plt.show()


# In[51]:


## Data Preprocessing


# In[52]:


##filling the missing values
data['Age'].fillna(data['Age'].median(), inplace=True) ## to fill the missing Age values we took the median 
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True) ## the most frequent value is used to fill the missing value 


# In[ ]:


##dropping unnecessary columns such as PassengerId,Ticket and Cabin


# In[56]:


data.drop(columns=['PassengerId','Ticket','Cabin'], inplace= True)


# In[57]:


data


# In[58]:


data.isnull().sum()


# In[61]:


## Outlier Handling
data = data[(np.abs(stats.zscore(data[['Age', 'Fare']])) < 3).all(axis=1)]


# In[62]:


data


# In[65]:


##Labeling
label_encoder = LabelEncoder()

# Apply label encoding to 'Sex' and 'Embarked'
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])


# In[67]:


data.shape


# In[71]:


## Test-Train Split
X= data.drop('Survived', axis=1) # features
y = data['Survived'] # targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Structure of the training sets and test sets \n")
X_train.shape,X_test.shape,y_train.shape,y_test.shape 


# In[72]:


## MODEL PREDICTION


# In[73]:


#RandomForestClassifier


# In[84]:


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf),'\n','-'*100)
print(classification_report(y_test, y_pred_rf))


# In[88]:


train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# In[82]:


confusionmat = confusion_matrix(y_test,y_pred_rf)


# In[83]:


print("Confusion matrix:", confusionmat)


# In[90]:


## K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn),'\n','-'*100)
print(classification_report(y_test, y_pred_knn))


# In[91]:


# SVC - Support Vector Classifier
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)

print("SVC Accuracy:", accuracy_score(y_test, y_pred_svc),'\n','-'*100)
print(classification_report(y_test, y_pred_svc))


# In[98]:


param_grid ={'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='accuracy',
                           n_jobs=-1, 
                           verbose=2)
grid_search.fit(X_train, y_train)


# In[97]:


print("Best Parameters for Random Forest:", grid_search.best_params_)
print("Best Accuracy Score:", grid_search.best_score_)


# In[99]:


# Cross Validation


# In[104]:





# In[112]:


rf_cv = RandomForestClassifier(**grid_search.best_params_)
rf_cv = cross_val_score(rf_cv, X_train, y_train, cv=5) ## for random forest
rf_cv


# In[113]:


print("Mean Cross-Validation Score:", np.mean(rf_cv))


# In[ ]:





# In[116]:


best_rf_model = grid_search.best_estimator_

# Training and test scores
train_score = best_rf_model.score(X_train, y_train)
test_score = best_rf_model.score(X_test, y_test)

print("Training Accuracy:", train_score *100)
print("Testing Accuracy:", test_score*100)


# In[ ]:





# In[ ]:





# In[ ]:




