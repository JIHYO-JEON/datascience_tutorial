# https://www.kaggle.com/subinium/subinium-tutorial-titanic-beginner

# Pandas and Numpy
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# Scikit-Learn
# linear regression, SVM, random forest, K-NN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# read csv : pd.read_csv('path/to/file')
train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')

print(train_df.info())
print(test_df.info())

# drop useless columns : df.drop(['column_name'], axis=1)
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_df = test_df.drop(['Name', 'Ticket'], axis=1)

print(train_df.head())
print(test_df.head())

print(train_df['Pclass'].value_counts())

# one-hot-encoding
'''
onehot = pd.get_dummies(df['column_name'])
onehot.columns = ['c1', 'c2']
df.drop([''], axis = 1)
df = df.join(onehot)
'''
# pclass
pclass_train_dummies = pd.get_dummies(train_df['Pclass'])
pclass_test_dummies = pd.get_dummies(test_df['Pclass'])
pclass_train_dummies.columns = ['first', 'second', 'third']
pclass_test_dummies.columns = ['first', 'second', 'third']
train_df.drop(['Pclass'], axis=1, inplace=True)
test_df.drop(['Pclass'], axis=1, inplace=True)
train_df = train_df.join(pclass_train_dummies)
test_df = test_df.join(pclass_test_dummies)

# sex
sex_train_dummies = pd.get_dummies(train_df['Sex'])
sex_test_dummies = pd.get_dummies(test_df['Sex'])
sex_train_dummies.columns = ['Female', 'Male']
sex_test_dummies.columns = ['Female', 'Male']
train_df.drop(['Sex'], axis=1, inplace=True)
test_df.drop(['Sex'], axis=1, inplace=True)
train_df = train_df.join(sex_train_dummies)
test_df = test_df.join(sex_test_dummies)

# Age
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(train_df['Age'].mean(), inplace=True)

# Fare
test_df['Fare'].fillna(0, inplace=True)

# Cabin
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)

# Embarked
print(train_df['Embarked'].value_counts())
train_df['Embarked'].fillna('S', inplace=True)
test_df['Embarked'].fillna('S', inplace=True)
embarked_train_dummies = pd.get_dummies(train_df['Embarked'])
embarked_test_dummies = pd.get_dummies(test_df['Embarked'])
train_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)
train_df = train_df.join(embarked_train_dummies)
test_df = test_df.join(embarked_test_dummies)

# Data Split
X_train = train_df.drop(['Survived'], axis=1) # input features
Y_train = train_df['Survived'] # output result
X_test = test_df.drop(["PassengerId"], axis=1).copy()
print(X_train.head())
print(X_test.head())
print(Y_train.head())

# ML algorithm
# Logistic Regression
logreg = LogisticRegression()
logreg = logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
print('Logistic Regression:', logreg.score(X_train, Y_train))

# Support Vector Machine
svc = SVC()
svc = svc.fit(X_train, Y_train)
# Y_pred = SVC.predict(X_test)
print('SVC:', svc.score(X_train, Y_train))

# Random Forest
random_forest = RandomForestClassifier()
random_forest = random_forest.fit(X_train, Y_train)
# Y_pred = random_forest.predict(X_test)
print('Random Forest:', random_forest.score(X_train, Y_train))

#KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(X_train, Y_train)
# Y_pred = knn.predict(X_test)
print('KNN:', knn.score(X_train, Y_train))

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)