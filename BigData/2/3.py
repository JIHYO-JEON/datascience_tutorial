import pandas as pd

# df = pd.read_csv('data/mtcars.csv')
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/mtcars.csv',index_col=0)
print(df.info())
print(df.head())

# Outlier
import numpy as np

q75, q50, q25 = np.percentile(df['wt'], [75, 50, 25])
iqr = q75 - q25
outlier_1 = df['wt'][df['wt'] >= q75 + 1.5*iqr]
outlier_2 = df['wt'][df['wt'] <= q25 - 1.5*iqr]
print(outlier_1.values, outlier_2.values)