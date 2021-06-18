import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

y_train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/y_train.csv',encoding='euc-kr')
X_train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/X_train.csv',encoding='euc-kr')
X_test = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/X_test.csv',encoding='euc-kr')

# Data Preprocessing
print(X_train.info())
print(X_train.head())
print(y_train.head())
print(X_test.info())
print(X_test.head())

# Check Null
print(X_train.isnull().sum())
print(X_test.isnull().sum())
X_train['환불금액'] = X_train['환불금액'].fillna(0)
X_test['환불금액'] = X_test['환불금액'].fillna(0)
print(X_train.isnull().sum())
print(X_test.isnull().sum())

# one-hot-encoding
dummies = pd.get_dummies(X_train['주구매상품'])
X_train = X_train.drop(['주구매상품'], axis=1)
X_train = X_train.join(dummies)
dummies = pd.get_dummies(X_test['주구매상품'])
X_test = X_test.drop(['주구매상품'], axis=1)
X_test = X_test.join(dummies)

dummies = pd.get_dummies(X_train['주구매지점'])
X_train = X_train.drop(['주구매지점'], axis=1)
X_train = X_train.join(dummies)
dummies = pd.get_dummies(X_test['주구매지점'])
X_test = X_test.drop(['주구매지점'], axis=1)
X_test = X_test.join(dummies)

# Drop
y_train = y_train.drop(['cust_id'], axis=1)

# LogisticRegression
LR = LogisticRegression()
LR = LR.fit(X_train, y_train)
print(LR.score(X_train, y_train))

# SVM
svc = SVC()
svc = svc.fit(X_train, y_train)
print(svc.score(X_train, y_train))

# Randomforest
rf = RandomForestClassifier()
rf = rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))

# KNN
knn = KNeighborsClassifier()
knn = knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))

print(X_train.shape)
print(X_test.shape)
Y_pred = knn.predict_proba(X_test)
submission = pd.DataFrame({
        "cust_id": X_test["cust_id"],
        "gender": Y_pred
    })
submission.to_csv('test.csv', index=False)