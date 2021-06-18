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

# Standardization
meanmean = df['qsec'].mean()
std = df['qsec'].std()
X = df['qsec'].apply(lambda x : (x-meanmean)/std)
print(X.min(), X.max())

# Outlier
import numpy as np

q75, q50, q25 = np.percentile(df['wt'], [75, 50, 25])
iqr = q75 - q25
outlier_1 = df['wt'][df['wt'] >= q75 + 1.5*iqr]
outlier_2 = df['wt'][df['wt'] <= q25 - 1.5*iqr]
print(outlier_1.values, outlier_2.values)

# Correlation
c = df.corr()['mpg']
print(c[1:].sort_values()[::-1])


# Add New Coloumn
index = list(df.index)
brand_list = [str(x).split(' ')[0] for x in index]
print(brand_list)
df['brand'] = pd.DataFrame(brand_list).values
print(df.head())