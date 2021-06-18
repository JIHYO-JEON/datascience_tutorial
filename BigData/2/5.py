import pandas as pd

# df = pd.read_csv('data/mtcars.csv')
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/mtcars.csv',index_col=0)
print(df.info())
print(df.head())

# Add New Coloumn
index = list(df.index)
brand_list = [str(x).split(' ')[0] for x in index]
print(brand_list)
df['brand'] = pd.DataFrame(brand_list).values
print(df.head())