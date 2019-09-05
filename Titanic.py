#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
train = pd.read_csv('train.csv')
train.describe()
train


# In[87]:


def clean_data(train):
    ## Recode names
    ## fix column names so the '-' character becomes '_'
    cols = train.columns
    train.columns = [str.replace('-', '_') for str in cols]
    
    ## Treat missing values
    ## Remove rows with missing values, accounting for mising values coded as '?'
    cols = ['PassengerId', 'Survived']
    for column in cols:
        gender_submission.loc[gender_submission[column] == '?', column] = np.nan
    gender_submission.dropna(axis = 0, inplace = True)

    ## Transform column data type
    ## Convert some columns to numeric values
    for column in cols:
        train[column] = pd.to_numeric(train[column])

    return train
gender_submission = clean_data(gender_submission)

print(train.columns)


# In[88]:


train.dtypes


# In[89]:


train.head(50)


# In[90]:


train['counts'] = 1
train[['Pclass','counts']].groupby('Pclass').count()


# In[91]:


train.describe()


# In[92]:


def count_unique(train, cols):
    for col in cols:
        print('\n' + 'For column ' + col)
        print(train[col].value_counts())

cat_cols = ['Survived', 'SibSp', 'Parch', 'Fare']
#count_unique(train, cat_cols)


# In[93]:


def plot_box(train, cols, col_x = 'Survived'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col_x, col, data=train)
        
        plt.xlabel(col_x) # Set text for the x axis
        plt.ylabel(col)# Set text for y axis
        plt.show()

num_cols = ['PassengerId', 'Pclass', 'Age',
            'Parch', 'Fare', 'counts']
plot_box(train, num_cols)


# In[94]:


def plot_box(train, cols, col_x = 'Survived'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col_x, col, data=train)
        
        plt.xlabel(col_x) # Set text for the x axis
        plt.ylabel(col)# Set text for y axis
        plt.show()

num_cols = ['PassengerId', 'Pclass', 'Age',
            'Parch', 'Fare', 'Survived']
plot_box(train, num_cols)


# In[95]:


def plot_density_hist(train, cols, bins = 30, hist = True):
    for col in cols:
        sns.set_style("whitegrid")
        sns.distplot(train[col], bins = bins, rug=True, hist = hist)
        plt.title('Histogram of ' + col) # Give the plot a main title
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel('Number of ' + col)# Set text for y axis
        plt.show()
        
plot_density_hist(train, num_cols) 


# In[96]:


train.columns = [str.replace('-', '_') for str in train.columns]


# In[229]:


(train.astype(np.object) == False).any()


# In[230]:


train.dtypes


# In[237]:


for col in train.columns:
    if train[col].dtype == object:
        count = 0
        count = [count + 1 for x in train[col] if x == '?']
        print(col + ' ' + str(sum(count)))


# In[232]:


cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']
for column in cols:
    train[column] = pd.to_numeric(train[column])
train[cols].dtypes


# In[106]:


train['Parch'].value_counts()


# In[107]:


train['Sex'].value_counts()


# In[108]:


train['SibSp'].value_counts()


# In[109]:


train['Age'].value_counts()


# In[111]:


train['Survived'].value_counts()


# In[113]:


print(train.shape)
print(train.PassengerId.unique().shape)


# In[114]:


train.to_csv('Prepared_train_DS.csv', index = False, header = True)


# In[118]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm


# In[123]:


survived_counts = train[['Survived','PassengerId']].groupby('Survived').count()
print(survived_counts)


# In[125]:


labels = np.array(train['Survived'])


# In[207]:


def encode_string(cat_features):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()



Features = encode_string(train['Sex'])


print(Features.shape)
print(Features[:, :]) 


# In[208]:


Features = np.concatenate([Features, np.array(train[['Parch', 'SibSp','Pclass']])], axis = 1)
print(Features.shape)
print(Features[:, :]) 


# In[209]:


nr.seed(9988)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 300)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])


# In[210]:


scaler = preprocessing.StandardScaler().fit(X_train[:,4:])
X_train[:,4:] = scaler.transform(X_train[:,4:])
X_test[:,4:] = scaler.transform(X_test[:,4:])
X_train[:2,]



logistic_mod = linear_model.LogisticRegression() 
logistic_mod.fit(X_train, y_train)


# In[211]:


print(logistic_mod.intercept_)
print(logistic_mod.coef_)


# In[212]:


probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])


# In[213]:


def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])
scores = score_model(probabilities, 0.5)
print(np.array(scores[:15]))
print(y_test[:15])


# In[214]:


def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])


    
print_metrics(y_test, scores)    


# In[218]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




