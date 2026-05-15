import tushare as ts
pro = ts.pro_api('**********************************')

# 获取ST股票列表
df = pro.stock_basic(
    exchange='',
    list_status='L',
    fields='ts_code,symbol,name,market,list_status'
)
print(df)
df.to_excel(r'C:\Users\kunbo\Desktop\tushare_data\股票st信息.xlsx',index=True)
# 名字里含ST的
st_df = df[df['name'].str.contains('ST', na=False)]
