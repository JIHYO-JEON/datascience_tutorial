import pandas as pd

# df = pd.read_csv('data/mtcars.csv')
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/mtcars.csv',index_col=0)
print(df.info())
print(df.head())

# Standardization
meanmean = df['qsec'].mean()
std = df['qsec'].std()
X = df['qsec'].apply(lambda x : (x-meanmean)/std)
print(X.min(), X.max())