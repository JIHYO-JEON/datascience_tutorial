import pandas as pd

# df = pd.read_csv('data/mtcars.csv')
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/mtcars.csv' ,index_col=0)
print(df.info())
print(df.head())


# Correlation
c = df.corr()['mpg']
print(c[1:].sort_values()[::-1])