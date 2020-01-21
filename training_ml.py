from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import numpy as np
from sklearn.model_selection import train_test_split
import glob
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

titanic_dataset = pd.read_csv('data/train.csv', thousands =',')

# Let's get best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


coloumn_list = ["PassengerId","Survived","Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
# coloumn_list = ["Survived","Pclass","Sex","Age","SibSp","Parch"]
# coloumn_list = ["Survived","Pclass","Sex","Age",]
n_cols = len(coloumn_list) -1

titanic_dataset = titanic_dataset[coloumn_list]

for column in coloumn_list:
    print('column',column,titanic_dataset[column].dtype)
    if titanic_dataset[column].dtype == type(object):
        titanic_dataset[column]  = [str(x) for x in titanic_dataset[column]]
        le = LabelEncoder()
        le.fit(titanic_dataset[column])
        titanic_dataset[column] =  tf.keras.utils.to_categorical(le.transform(titanic_dataset[column]))

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
                    return 37
        elif Pclass == 2:
                    return 29
        else:
                    return 24
    else:
            return Age
    
titanic_dataset['Age'] = titanic_dataset[['Age','Pclass']].apply(impute_age,axis=1)

def has_family(cols):
    SibSp = cols[0]
    Parch = cols[1]
    
    if SibSp > 0 or Parch > 0:
        return 1
    else:
        return 0

titanic_dataset['Family'] = titanic_dataset[['SibSp','Parch']].apply(has_family,axis=1)

X = titanic_dataset[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked","Family"]]#independent columns
y = titanic_dataset[["Survived"]] #target column i.e price range#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features

# prib

titanic_dataset = titanic_dataset[["Fare","Survived","Sex","Pclass","Age","Family"]]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(titanic_dataset, titanic_dataset["Survived"]):
    titanic_train_set = titanic_dataset.loc[train_index]
    titanic_test_set = titanic_dataset.loc[test_index]

X = titanic_train_set.drop(["Survived"],axis=1)
Y =  tf.keras.utils.to_categorical(titanic_train_set["Survived"].copy())

test_x = titanic_test_set.drop(["Survived"],axis=1)
test_y = tf.keras.utils.to_categorical( titanic_test_set["Survived"].copy())

from sklearn.ensemble import RandomForestClassifier

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 2000, random_state = 42)

# print(X.head(100))
# Train the model on training data
rf.fit(X, Y)

pred = rf.predict(test_x)

# print('results', pred , test_y)


from sklearn.model_selection import cross_val_predict
y_train_pred_randomforest = cross_val_predict(rf, X, Y, cv=3)


print("cross val with Random forest")
from sklearn.model_selection import cross_val_score
# print(y_train_pred_randomforest)
print(cross_val_score(rf, X, Y, cv=3, scoring="accuracy"))


#Logistic regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

Y_log =  titanic_train_set["Survived"].copy()

logmodel.fit(X,Y_log)

print(cross_val_score(logmodel, X, Y_log, cv=3, scoring="accuracy"))