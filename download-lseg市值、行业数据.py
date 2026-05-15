import lseg.data as ld
import pandas as pd
import datetime
import time
ld.open_session()

def calculate_day(period):
    """
    period: 距今多少天作为结束日
    返回 [start, end]，窗口为整整一年（按年偏移，不是365天）
    """
    end_date = datetime.date.today() - datetime.timedelta(days=period)
    # 用 replace(year=...) 精确回退一年，自动处理闰年
    start_date = end_date.replace(year=end_date.year - 4)
    return [start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]


def get_fundamental(codes, fields, period, batch_size=500):
    all_batches = []

    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        print(f"处理批次 {i // batch_size + 1}：{len(batch)} 个 RIC...")

        df_batch = ld.get_history(
            universe=batch,
            fields=fields,
            start=calculate_day(period)[0],
            end=calculate_day(period)[1],
            interval="1M"
        )
        # 批次内去重
        df_batch = df_batch[~df_batch.index.duplicated(keep='first')]
        all_batches.append(df_batch)
        result = pd.concat(all_batches, axis=1, sort=False)
    return result


def get_industry_info(codes, batch_size=300, sleep_time=0.8):
    """
    获取A股的行业信息（TRBC行业分类）
    - codes: RIC列表
    - batch_size: 每批请求数量（行业字段通常可以设大一点）
    - sleep_time: 每批间隔时间（防限流）
    """
    print("开始获取行业信息...")
    all_dfs = []

    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        df_batch = ld.get_data(
            universe=batch,
            fields=[
                "TR.CommonName",  # 公司名称
                "TR.CompanyName",
                "TR.TRBCIndustry",  # TRBC 行业（推荐）
                "TR.TRBCIndustryGroup",  # TRBC 行业组
                "TR.TRBCBusinessSector",  # TRBC 商业部门
                "TR.GICSSector",  # GICS 一级行业（可选）
            ]
        )
        all_dfs.append(df_batch)
        time.sleep(sleep_time)

    industry_df = pd.concat(all_dfs, ignore_index=True)
    print(f"行业信息获取完成！共 {len(industry_df)} 条记录")
    return industry_df



# 读取股票池
universe = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.ExchangeMarketIdCode,"XSHG","XSHE"), CURN=CNY)'
fields = [
#    "TR.F.GrossProfMarg",
    "TR.CompanyMarketCap",
#    "TR.F.NetCashFlowOp",
#    "TR.F.NetIncAfterTax",
#    "TR.F.EV",
#    "TR.EBITDA"
]
df_rics = ld.get_data(universe=universe, fields="TR.RIC")
rics = df_rics['RIC'].tolist()
print(f"成功获取 {len(df_rics)} 支股票")


# 2. 获取行业信息（调用我们封装的函数）
#industry_df = get_industry_info(codes=rics, batch_size=300, sleep_time=0.8)

# 保存行业信息
#industry_df.to_excel(r"C:\Users\kunbo\Desktop\lseg\因子研究\Industry_Info.xlsx",   index=False)

for finfac in fields:
    selected_stock = get_fundamental(codes=rics, fields=finfac, period=120)
    print(selected_stock)

    filename = finfac.replace('TR.', '').replace('.', '_')
    selected_stock.to_excel(rf"C:\Users\kunbo\Desktop\lseg\因子研究\{filename}.xlsx", index=True)
    print(f"已保存：{filename}.xlsx")
