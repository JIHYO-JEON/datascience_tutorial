import pandas as pd

# df = pd.read_csv('data/mtcars.csv')
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/mtcars.csv',index_col=0)
print(df.info())
print(df.head())

# min-max scaler
maxmax = df['qsec'].max()
minmin = df['qsec'].min()
X = df['qsec'].apply(lambda x : (x-minmin)/(maxmax-minmin))
print(len(X[X>0.5]))