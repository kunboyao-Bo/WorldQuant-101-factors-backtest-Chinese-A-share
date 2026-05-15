import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df1=pd.read_excel(r'C:\Users\kunbo\Desktop\tushare_data\adj_factor_pivot_22-25_1.xlsx',index_col=0)
df2=pd.read_excel(r'C:\Users\kunbo\Desktop\tushare_data\adj_factor_pivot_22-25_2.xlsx',index_col=0)
#df3=pd.read_excel(r'C:\Users\kunbo\Desktop\tushare_data\adj_factor_pivot_22-25_3.xlsx',index_col=0)

result = pd.concat([df1, df2], axis=0, join='outer', ignore_index=False)

result = result.sort_index(ascending=True)
print(result.iloc[:, :3])
result.to_excel(r'C:\Users\kunbo\Desktop\tushare_data\adj_factor_pivot_22-25.xlsx')