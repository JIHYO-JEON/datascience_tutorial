import pandas as pd

y_train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/y_train.csv',encoding='euc-kr')
X_train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/X_train.csv',encoding='euc-kr')
X_test = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/X_test.csv',encoding='euc-kr')

# Merge Train Data
train_df = pd.merge(y_train, X_train)
print(train_df.head())
print(train_df.info())

# check data
print(train_df.isnull().sum())
print(X_test.isnull().sum())

print(train_df['환불금액'].describe())
print(X_test['환불금액'].describe())

train_df['환불금액'] = train_df['환불금액'].fillna(0)
X_test['환불금액'] = X_test['환불금액'].fillna(0)
print(train_df.isnull().sum())
print(X_test.isnull().sum())

print(train_df.info())
print(train_df['주구매상품'])

print(train_df['주구매상품'].value_counts().index)
print(train_df['주구매지점'].value_counts().index)

from sklearn.model_selection import train_test_split

x = train_df.drop(['cust_id', 'gender'], axis = 1)
test = X_test.drop(['cust_id'], axis = 1)

print(x.head())
y = train_df['gender']
print(y.head())

x_dum = pd.get_dummies(x)
print(x_dum)
feature_name_lst = x_dum.columns
test_dum = pd.get_dummies(test)