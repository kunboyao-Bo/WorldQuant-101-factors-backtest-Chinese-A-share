import tushare as ts
import time
import pandas as pd
from datetime import datetime, timedelta
import os

# 配置
ts.set_token('*****************')    #
pro = ts.pro_api()



save_dir = r'C:\Users\kunbo\Desktop\tushare_data'
os.makedirs(save_dir, exist_ok=True)
trade_dates = pd.read_excel(r'C:\Users\kunbo\Desktop\tushare_data\日期.xlsx', header=None)[0]
#trade_dates = ['20250327']
dates = pd.to_datetime(trade_dates).dt.strftime('%Y%m%d').tolist()
#dates = pd.Series(trade_dates).tolist()
output_file = os.path.join(save_dir, 'adj_factor_pivot_22-25_2.xlsx')  # pivot版文件名

all_data = []
skipped_dates = []

for i, date in enumerate(dates):
    print(f"[{i + 1}/{len(dates)}] 尝试 {date} ...")
    try:
        #df = pro.daily(trade_date=date, fields='ts_code,trade_date,adj_factor')
        df = pro.adj_factor(trade_date=date, fields='ts_code,trade_date,adj_factor')
        if not df.empty:
            all_data.append(df)
            print(f"  成功 {len(df)} 条")
        else:
            print("  跳过（非交易日）")
            skipped_dates.append(date)          # ← 这里记录
    except Exception as e:
        print(f"  出错: {e}")
        time.sleep(10)
        continue
    time.sleep(4)

# 循环结束后打印或保存跳过的日期
print("\n跳过的日期（非交易日或其他无数据日）：")
print(skipped_dates)
print(f"共跳过 {len(skipped_dates)} 天")

# 处理 & pivot
if all_data:
    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all[['ts_code', 'trade_date', 'adj_factor']]
    df_all['trade_date'] = pd.to_datetime(df_all['trade_date'], format='%Y%m%d')  # 转日期格式，便于排序
    df_all.sort_values(['ts_code', 'trade_date'], inplace=True)

    # Pivot 成宽表
    df_pivot = df_all.pivot(index='trade_date', columns='ts_code', values='adj_factor')
    df_pivot = df_pivot.sort_index(axis=1)  # 日期列升序

    df_pivot.to_excel(output_file, index=True)
    print(f"\nPivot 完成！保存到 {output_file}")
    print(f"交易日数（列数）：{df_pivot.shape[1]}")
    print("前5股票前5交易日预览：")
    print(df_pivot.iloc[:5, :5])
else:
    print("无数据")

print("重新跑")
print(skipped_dates)