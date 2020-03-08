
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
# import tensorflow as tf
# from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

pd.set_option('display.max_rows', 500)

titanic_dataset = pd.read_csv('data/train.csv', thousands=',')


# Let's get best features


coloumn_list = ["PassengerId", "Survived", "Pclass", "Sex",
                "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

n_cols = len(coloumn_list) - 1

titanic_dataset = titanic_dataset[coloumn_list]

titanic_dataset['Sex'] = titanic_dataset['Sex'].map({'male': 0, 'female': 1})


def column_encoder(coloumn_list, ds):
    for column in coloumn_list:
        # print('column',column,titanic_dataset[column].dtype)
        if ds[column].dtype == type(object):
            ds[column] = [str(x) for x in ds[column]]
            le = LabelEncoder()
            le.fit(ds[column])
            # tf.keras.utils.to_categorical(le.transform(ds[column]))
            ds[column] = pd.get_dummies(ds[column])
    return ds


def column_to_categorical(ds, col):
    ds[col] = pd.get_dummies(ds[col])
    return ds


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


def has_family(cols):
    SibSp = cols[0]
    Parch = cols[1]

    if SibSp > 0 or Parch > 0:
        return 1
    else:
        return 0


def assign_age_group(cols):
    Age = cols[0]
    Sex = cols[1]
    AgeGroup = 1

    if Sex == 'female1':
        if Age > 10 and Age < 20:
            AgeGroup = 2
        # elif Age >20 and Age <40:
        #     AgeGroup = 3
        elif Age > 40:
            AgeGroup = 4
    else:
        if Age > 10 and Age < 20:
            AgeGroup = 2
        elif Age > 20 and Age < 40:
            AgeGroup = 3
        elif Age > 40:
            AgeGroup = 4

    return AgeGroup


ageDiscretizer = KBinsDiscretizer(
    n_bins=4, encode='onehot', strategy='uniform')


def age_KBinsDiscretizer(ds):
    ageDiscretizer.fit([ds['Age']])
    z = ageDiscretizer.transform([ds['Age']])
    return z


titanic_dataset['Age'] = titanic_dataset[[
    'Age', 'Sex']].apply(assign_age_group, axis=1)
# titanic_dataset['Age'].fillna((titanic_dataset['Age'].mean()), inplace=True)

titanic_dataset = column_encoder(coloumn_list, titanic_dataset)
# titanic_dataset[['Age','Pclass']].apply(impute_age,axis=1)
titanic_dataset = column_to_categorical(titanic_dataset, 'Pclass')
# titanic_dataset =  column_to_categorical(titanic_dataset,'Age')
# titanic_dataset =  column_to_categorical(titanic_dataset,'Embarked')
titanic_dataset['Family'] = titanic_dataset[[
    'SibSp', 'Parch']].apply(has_family, axis=1)


X = titanic_dataset[["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch",
                     "Ticket", "Fare", "Cabin", "Embarked", "Family"]]  # independent columns
# target column i.e price range#apply SelectKBest class to extract top 10 best features
y = titanic_dataset[["Survived"]]

# X = data set for training


def select_best_features(X):

    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(10, 'Score'))  # print 10 best features


# new dataset only with best features
titanic_dataset = titanic_dataset[[
    "Fare", "Survived", "Sex", "Pclass", "Age", "Family", "Embarked"]]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(titanic_dataset, titanic_dataset["Survived"]):
    titanic_train_set = titanic_dataset.loc[train_index]
    titanic_test_set = titanic_dataset.loc[test_index]

X = titanic_train_set.drop(["Survived"], axis=1)
# tf.keras.utils.to_categorical(titanic_train_set["Survived"].copy())
Y = titanic_train_set["Survived"].copy()

test_x = titanic_test_set.drop(["Survived"], axis=1)
# tf.keras.utils.to_categorical( titanic_test_set["Survived"].copy())
test_y = titanic_test_set["Survived"].copy()


def StandardizeData(data_frame):

    scl = StandardScaler()
    scl = scl.fit(data_frame)
    std_features = scl.transform(data_frame)

    data_frame = pd.DataFrame(std_features, columns=data_frame.columns)
    return data_frame

# X = StandardizeData(X)
# test_x = StandardizeData(test_x)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

# pkl_filename = "pickle_model_1.pkl"


def pickle_model(pkl_filename):

    with open(pkl_filename, 'wb') as file:
        pickle.dump(rf, file)


def read_pickle_model(pkl_filename):
    with open(pkl_filename, 'rb') as file:
        return pickle.load(file)


# Instantiate model with 1000 decision trees
rf = XGBClassifier()

rf.fit(X, Y)

# print("cross val with Random forest")
# y_train_pred_randomforest = cross_val_predict(rf, X, Y, cv=3)

# print('cross_val_score', cross_val_score(rf, X, Y, cv=3, scoring="accuracy"))

result_y = rf.predict(test_x)

result_y = [round(a) for a in result_y]

print('Accuracy', accuracy_score(test_y, result_y))
print('precision_score', precision_score(test_y, result_y))
print('recall_score', recall_score(test_y, result_y))
print('roc_auc_score', roc_auc_score(test_y, result_y))


# Lets' predict with test set

titanic_pdataset = pd.read_csv('data/test.csv', thousands=',')

coloumn_list = ["PassengerId", "Pclass", "Sex", "Age",
                "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

# titanic_pdataset['Age'].fillna((titanic_pdataset['Age'].mean()), inplace=True)
# titanic_pdataset['Age'] = age_KBinsDiscretizer(titanic_dataset)
# titanic_pdataset['Fare'] = 0
titanic_dataset['Sex'] = titanic_dataset['Sex'].map({'male': 0, 'female': 1})
titanic_pdataset['Age'] = titanic_pdataset[[
    'Age', 'Sex']].apply(assign_age_group, axis=1)
titanic_pdataset = column_encoder(coloumn_list, titanic_pdataset)
titanic_pdataset = column_to_categorical(titanic_pdataset, 'Pclass')
# titanic_dataset =  column_to_categorical(titanic_dataset,'Age')

# titanic_pdataset['Age'] = titanic_pdataset[['Age','Pclass']].apply(impute_age,axis=1)
titanic_pdataset['Family'] = titanic_pdataset[[
    'SibSp', 'Parch']].apply(has_family, axis=1)
titanic_pdataset.fillna(0, inplace=True)

df = titanic_pdataset[["Fare", "Sex", "Pclass", "Age", "Family", "Embarked"]]
# df = StandardScaler(titanic_pdataset[["Fare","Sex","Pclass","Age","Family","Embarked"]])

results = rf.predict(df)

test_results = pd.DataFrame(
    {'PassengerId': titanic_pdataset['PassengerId'], 'Survived': results})
test_results.to_csv('data/results_xgb.csv', index=False)
