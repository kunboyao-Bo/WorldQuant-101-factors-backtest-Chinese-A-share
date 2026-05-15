
import pandas as pd
import numpy as np
'''
WorldQuant 101 所有因子在论文原始定义里就已经是截面 rank 归一化的，量纲天然一致，不需要你额外做 z-score。
将所有的比较全部改为做差，用于取得离散的因子值，而非bool
筛选条件
IC均值绝对值 > 0.02 ,   可提高到 > 0.025 或 > 0.03（太低容易过拟合）
ICIR绝对值 > 0.3,    建议提高到 > 0.5（周频/月频更推荐 >0.6）
IC t统计量显著,      建议同时要求样本量足够（>50期）
分组严格单调性,     
Q5-Q1 正且稳定,


'''

# ── 注册表：每个alpha需要哪些df ──────────────────────────────
ALPHA_REGISTRY = {
    "alpha001": {"func": lambda dfs: alpha001(dfs["close"]),"required": ["close"]},
    "alpha002": {"func": lambda dfs: alpha002(dfs["close"], dfs["open"], dfs["volume"]),"required": ["close", "open", "volume"]},
    "alpha003": {"func": lambda dfs: alpha003(dfs["open"], dfs["volume"]),"required": ["open", "volume"]},
    "alpha004": {"func": lambda dfs: alpha004(dfs["low"]),"required": ["low"]},
    "alpha005": {"func": lambda dfs: alpha005(dfs["open"], dfs["vwap"], dfs["close"]),"required": ["open", "vwap", "close"]},
    "alpha006": {"func": lambda dfs: alpha006(dfs["open"], dfs["volume"]),"required": ["open", "volume"]},
    "alpha007": {"func": lambda dfs: alpha007(dfs["close"], dfs["volume"]),"required": ["close", "volume"]},
    "alpha008": {"func": lambda dfs: alpha008(dfs["close"], dfs["open"]),"required": ["close", "open"]},
    "alpha009": {"func": lambda dfs: alpha009(dfs["close"]),"required": ["close"]},
    "alpha010": {"func": lambda dfs: alpha010(dfs["close"]),"required": ["close"]},
    "alpha011": {"func": lambda dfs: alpha011(dfs["close"],dfs["vwap"],dfs["volume"]),"required": ["close","vwap","volume"]},
    "alpha012": {"func": lambda dfs: alpha012(dfs["close"],dfs["volume"]),"required": ["close","volume"]},
    "alpha013": {"func": lambda dfs: alpha013(dfs["close"], dfs["volume"]),"required": ["close", "volume"]},
    "alpha014": {"func": lambda dfs: alpha014(dfs["close"], dfs["open"],  dfs["volume"]),"required": ["close","open", "volume"]},
    "alpha015": {"func": lambda dfs: alpha015(dfs["high"], dfs["volume"]),"required": ["high","volume"]},
    "alpha016": {"func": lambda dfs: alpha016(dfs["high"], dfs["volume"]),"required": ["high","volume"]},
    "alpha017": {"func": lambda dfs: alpha017(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha018": {"func": lambda dfs: alpha018(dfs["close"], dfs["open"]), "required": ["close", "open"]},
    "alpha019": {"func": lambda dfs: alpha019(dfs["close"]), "required": ["close"]},
    "alpha020": {"func": lambda dfs: alpha020(dfs["close"], dfs["open"], dfs["high"], dfs["low"]),"required": ["close", "open", "high", "low"]},
    "alpha021": {"func": lambda dfs: alpha021(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha022": {"func": lambda dfs: alpha022(dfs["close"], dfs["high"], dfs["volume"]),"required": ["close", "high", "volume"]},
    "alpha023": {"func": lambda dfs: alpha023(dfs["high"]), "required": ["high"]},
    "alpha024": {"func": lambda dfs: alpha024(dfs["close"]), "required": ["close"]},
    "alpha025": {"func": lambda dfs: alpha025(dfs["close"], dfs["high"], dfs["volume"], dfs["vwap"]),"required": ["close", "high", "volume", "vwap"]},
    "alpha026": {"func": lambda dfs: alpha026(dfs["high"], dfs["volume"]), "required": ["high", "volume"]},
    "alpha027": {"func": lambda dfs: alpha027(dfs["volume"], dfs["vwap"]), "required": ["volume", "vwap"]},
    "alpha028": {"func": lambda dfs: alpha028(dfs["close"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["close", "high", "low", "volume"]},
    "alpha029": {"func": lambda dfs: alpha029(dfs["close"]), "required": ["close"]},
    "alpha030": {"func": lambda dfs: alpha030(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha031": {"func": lambda dfs: alpha031(dfs["close"], dfs["low"], dfs["volume"]), "required": ["close", "low", "volume"]},
    "alpha032": {"func": lambda dfs: alpha032(dfs["close"], dfs["vwap"]), "required": ["close", "vwap"]},
    "alpha033": {"func": lambda dfs: alpha033(dfs["close"], dfs["open"]), "required": ["close", "open"]},
    "alpha034": {"func": lambda dfs: alpha034(dfs["close"]), "required": ["close"]},
    "alpha035": {"func": lambda dfs: alpha035(dfs["close"], dfs["high"], dfs["low"], dfs["volume"]),"required": ["close", "high", "low", "volume"]},
    "alpha036": {"func": lambda dfs: alpha036(dfs["close"], dfs["open"], dfs["volume"], dfs["vwap"]),"required": ["close", "open", "volume", "vwap"]},
    "alpha037": {"func": lambda dfs: alpha037(dfs["close"], dfs["open"]), "required": ["close", "open"]},
    "alpha038": {"func": lambda dfs: alpha038(dfs["close"], dfs["open"]), "required": ["close", "open"]},
    "alpha039": {"func": lambda dfs: alpha039(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha040": {"func": lambda dfs: alpha040(dfs["high"], dfs["volume"]), "required": ["high", "volume"]},
    "alpha041": {"func": lambda dfs: alpha041(dfs["high"], dfs["low"], dfs["vwap"]), "required": ["high", "low", "vwap"]},
    "alpha042": {"func": lambda dfs: alpha042(dfs["close"], dfs["vwap"]), "required": ["close", "vwap"]},
    "alpha043": {"func": lambda dfs: alpha043(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha044": {"func": lambda dfs: alpha044(dfs["high"], dfs["volume"]), "required": ["high", "volume"]},
    "alpha045": {"func": lambda dfs: alpha045(dfs["close"], dfs["volume"]), "required": ["close", "volume"]},
    "alpha046": {"func": lambda dfs: alpha046(dfs["close"]), "required": ["close"]},
    "alpha047": {"func": lambda dfs: alpha047(dfs["high"], dfs["close"], dfs["volume"], dfs["vwap"]), "required": ["high", "close", "volume", "vwap"]},
    "alpha048": {"func": lambda dfs: alpha048(dfs["close"], dfs["industry"]), "required": ["close", "industry"]},
    "alpha049": {"func": lambda dfs: alpha049(dfs["close"]), "required": ["close"]},
    "alpha050": {"func": lambda dfs: alpha050(dfs["volume"], dfs["vwap"]), "required": ["volume", "vwap"]},
    "alpha051": {"func": lambda dfs: alpha051(dfs["close"]), "required": ["close"]},
    "alpha052": {"func": lambda dfs: alpha052(dfs["low"], dfs["volume"], dfs["close"]), "required": ["low", "volume", "close"]},
    "alpha053": {"func": lambda dfs: alpha053(dfs["close"], dfs["low"], dfs["high"]), "required": ["close", "low", "high"]},
    "alpha054": {"func": lambda dfs: alpha054(dfs["close"], dfs["low"], dfs["high"], dfs["open"]), "required": ["close", "low", "high", "open"]},
    "alpha055": {"func": lambda dfs: alpha055(dfs["close"], dfs["low"], dfs["high"], dfs["volume"]), "required": ["close", "low", "high", "volume"]},
    "alpha056": {"func": lambda dfs: alpha056(dfs["close"], dfs["cap"]), "required": ["close", "cap"]},
    "alpha057": {"func": lambda dfs: alpha057(dfs["close"], dfs["vwap"]), "required": ["close", "vwap"]},
    "alpha058": {"func": lambda dfs: alpha058(dfs["vwap"], dfs["volume"], dfs["industry"]), "required": ["vwap", "volume", "industry"]},
    "alpha059": {"func": lambda dfs: alpha059(dfs["vwap"], dfs["volume"], dfs["industry"]), "required": ["vwap", "volume", "industry"]},
    "alpha060": {"func": lambda dfs: alpha060(dfs["close"], dfs["low"], dfs["high"], dfs["volume"]), "required": ["close", "low", "high", "volume"]},
    "alpha061": {"func": lambda dfs: alpha061(dfs["vwap"], dfs["volume"]), "required": ["vwap", "volume"]},
    "alpha062": {"func": lambda dfs: alpha062(dfs["vwap"], dfs["open"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["vwap", "open", "high", "low", "volume"]},
    "alpha063": {"func": lambda dfs: alpha063(dfs["close"], dfs["vwap"], dfs["open"], dfs["volume"], dfs["industry"]), "required": ["close", "vwap", "open", "volume", "industry"]},
    "alpha064": {"func": lambda dfs: alpha064(dfs["open"], dfs["low"], dfs["high"], dfs["vwap"], dfs["volume"]), "required": ["open", "low", "high", "vwap", "volume"]},
    "alpha065": {"func": lambda dfs: alpha065(dfs["open"], dfs["vwap"], dfs["volume"]), "required": ["open", "vwap", "volume"]},
    "alpha066": {"func": lambda dfs: alpha066(dfs["low"], dfs["vwap"], dfs["open"], dfs["high"]), "required": ["low", "vwap", "open", "high"]},
    "alpha067": {"func": lambda dfs: alpha067(dfs["high"], dfs["vwap"], dfs["volume"], dfs["industry"]), "required": ["high", "vwap", "volume", "industry"]},
    "alpha068": {"func": lambda dfs: alpha068(dfs["high"], dfs["close"], dfs["low"], dfs["volume"]), "required": ["high", "close", "low", "volume"]},
    "alpha069": {"func": lambda dfs: alpha069(dfs["close"], dfs["vwap"], dfs["volume"], dfs["industry"]), "required": ["close", "vwap", "volume", "industry"]},
    "alpha070": {"func": lambda dfs: alpha070(dfs["close"], dfs["vwap"], dfs["volume"], dfs["industry"]), "required": ["close", "vwap", "volume", "industry"]},
    "alpha071": {"func": lambda dfs: alpha071(dfs["close"], dfs["low"], dfs["open"], dfs["vwap"], dfs["volume"]), "required": ["close", "low", "open", "vwap", "volume"]},
    "alpha072": {"func": lambda dfs: alpha072(dfs["high"], dfs["low"], dfs["vwap"], dfs["volume"]), "required": ["high", "low", "vwap", "volume"]},
    "alpha073": {"func": lambda dfs: alpha073(dfs["open"], dfs["low"], dfs["vwap"]), "required": ["open", "low", "vwap"]},
    "alpha074": {"func": lambda dfs: alpha074(dfs["close"], dfs["high"], dfs["vwap"], dfs["volume"]), "required": ["close", "high", "vwap", "volume"]},
    "alpha075": {"func": lambda dfs: alpha075(dfs["vwap"], dfs["low"], dfs["volume"]), "required": ["vwap", "low", "volume"]},
    "alpha076": {"func": lambda dfs: alpha076(dfs["low"], dfs["vwap"], dfs["volume"], dfs["industry"]), "required": ["low", "vwap", "volume", "industry"]},
    "alpha077": {"func": lambda dfs: alpha077(dfs["high"], dfs["low"], dfs["vwap"], dfs["volume"]), "required": ["high", "low", "vwap", "volume"]},
    "alpha078": {"func": lambda dfs: alpha078(dfs["low"], dfs["vwap"], dfs["volume"]), "required": ["low", "vwap", "volume"]},
    "alpha079": {"func": lambda dfs: alpha079(dfs["close"], dfs["open"], dfs["vwap"], dfs["volume"], dfs["industry"]), "required": ["close", "open", "vwap", "volume", "industry"]},
    "alpha080": {"func": lambda dfs: alpha080(dfs["open"], dfs["high"], dfs["volume"], dfs["industry"]), "required": ["open", "high", "volume", "industry"]},
    "alpha081": {"func": lambda dfs: alpha081(dfs["vwap"], dfs["volume"]), "required": ["vwap", "volume"]},
    "alpha082": {"func": lambda dfs: alpha082(dfs["open"], dfs["volume"], dfs["industry"]), "required": ["open", "volume", "industry"]},
    "alpha083": {"func": lambda dfs: alpha083(dfs["high"], dfs["low"], dfs["close"], dfs["vwap"], dfs["volume"]), "required": ["high", "low", "close", "vwap", "volume"]},
    "alpha084": {"func": lambda dfs: alpha084(dfs["vwap"], dfs["close"]), "required": ["vwap", "close"]},
    "alpha085": {"func": lambda dfs: alpha085(dfs["high"], dfs["close"], dfs["low"], dfs["volume"]), "required": ["high", "close", "low", "volume"]},
    "alpha086": {"func": lambda dfs: alpha086(dfs["close"], dfs["open"], dfs["vwap"], dfs["volume"]), "required": ["close", "open", "vwap", "volume"]},
    "alpha087": {"func": lambda dfs: alpha087(dfs["close"], dfs["vwap"], dfs["volume"], dfs["industry"]), "required": ["close", "vwap", "volume", "industry"]},
    "alpha088": {"func": lambda dfs: alpha088(dfs["open"], dfs["low"], dfs["high"], dfs["close"], dfs["volume"]), "required": ["open", "low", "high", "close", "volume"]},
    "alpha089": {"func": lambda dfs: alpha089(dfs["low"], dfs["vwap"], dfs["volume"], dfs["industry"]), "required": ["low", "vwap", "volume", "industry"]},
    "alpha090": {"func": lambda dfs: alpha090(dfs["close"], dfs["low"], dfs["volume"], dfs["industry"]), "required": ["close", "low", "volume", "industry"]},
    "alpha091": {"func": lambda dfs: alpha091(dfs["close"], dfs["vwap"], dfs["volume"], dfs["industry"]), "required": ["close", "vwap", "volume", "industry"]},
    "alpha092": {"func": lambda dfs: alpha092(dfs["high"], dfs["low"], dfs["close"], dfs["open"], dfs["volume"]), "required": ["high", "low", "close", "open", "volume"]},
    "alpha093": {"func": lambda dfs: alpha093(dfs["vwap"], dfs["close"], dfs["volume"], dfs["industry"]), "required": ["vwap", "close", "volume", "industry"]},
    "alpha094": {"func": lambda dfs: alpha094(dfs["vwap"], dfs["volume"]), "required": ["vwap", "volume"]},
    "alpha095": {"func": lambda dfs: alpha095(dfs["open"], dfs["high"], dfs["low"], dfs["volume"]), "required": ["open", "high", "low", "volume"]},
    "alpha096": {"func": lambda dfs: alpha096(dfs["vwap"], dfs["close"], dfs["volume"]), "required": ["vwap", "close", "volume"]},
    "alpha097": {"func": lambda dfs: alpha097(dfs["low"], dfs["vwap"], dfs["volume"], dfs["industry"]), "required": ["low", "vwap", "volume", "industry"]},
    "alpha098": {"func": lambda dfs: alpha098(dfs["vwap"], dfs["open"], dfs["volume"]), "required": ["vwap", "open", "volume"]},
    "alpha099": {"func": lambda dfs: alpha099(dfs["high"], dfs["low"], dfs["close"], dfs["volume"]), "required": ["high", "low", "close", "volume"]},
    "alpha100": {"func": lambda dfs: alpha100(dfs["close"], dfs["high"], dfs["low"], dfs["volume"], dfs["industry"]), "required": ["close", "high", "low", "volume", "industry"]},
    "alpha101": {"func": lambda dfs: alpha101(dfs["close"], dfs["open"], dfs["high"], dfs["low"]), "required": ["close", "open", "high", "low"]},
    # 继续注册...
}
def indneutralize(factor_panel, industry_s):
    """
    截面行业中性化（向量化版本）：每个交易日减去同行业截面均值。
    industry_s: index=ts_code, values=行业标签字符串
    """
    common_stocks = factor_panel.columns.intersection(industry_s.index)
    panel = factor_panel[common_stocks]          # (date, stock)
    ind   = industry_s.reindex(common_stocks)    # (stock,) 行业标签

    # 每个行业的截面均值：(date, industry)
    # stack → groupby行业 → mean → unstack 回 (date, stock)
    ind_mean = (
        panel.T                                  # (stock, date)
             .groupby(ind)                       # 按行业分组
             .transform('mean')                  # 同行业内广播均值
             .T                                  # (date, stock)
    )

    neutralized = factor_panel.copy()
    neutralized[common_stocks] = panel - ind_mean
    return neutralized

def alpha001(close: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha#1:rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
    """
    #日收益率
    returns = close.pct_change()
    #20日滚动标准差
    stddev_20 = returns.rolling(20).std()
    #条件替换：returns < 0 用 stddev_20，否则用 close
    inner = pd.DataFrame(
        np.where(returns < 0, stddev_20, close),
        index=close.index,
        columns=close.columns
    )
    #SignedPower(x, 2) = sign(x) * |x|^2 = x * |x|（保号平方）
    signed_power = inner * inner.abs()
    #Ts_ArgMax(..., 5)：过去5日中最大值出现在第几天（0-indexed from window start）
    #    rolling argmax：在长度5的窗口内找最大值的相对位置
    def rolling_argmax(df, window=5):
        result = df.rolling(window).apply(lambda x: np.argmax(x), raw=True)
        return result

    ts_argmax = rolling_argmax(signed_power, window=5)
    # 截面 rank（升序，归一化到 [0,1]），再减 0.5
    alpha = ts_argmax.rank(axis=1, pct=True) - 0.5
    return alpha

def alpha002(close, open, volume):
    """
    Alpha#02: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))

    delta(x, d)       = x - x.shift(d)
    correlation(x, y, d) = 过去d天的滚动相关系数
    rank(x)           = 截面百分位排名
    """
    # delta(log(volume), 2)
    log_vol = np.log(volume)
    delta_log_vol = log_vol - log_vol.shift(2)

    # (close - open) / open
    ret_co = (close - open) / open

    # 截面 rank（每行按股票排名）
    rank_delta = delta_log_vol.rank(axis=1, pct=True)
    rank_ret = ret_co.rank(axis=1, pct=True)

    # 滚动6期相关系数（时序，每只股票独立）
    # 用 rolling corr：对每只股票，rank_delta和rank_ret在时间轴上的6期相关
    alpha = -1 * rank_delta.rolling(6).corr(rank_ret)
    return alpha

def alpha003(open, volume):
    """
    Alpha#03:  (-1 * correlation(rank(open), rank(volume), 10))
    correlation(x, y, d) = 过去d天的滚动相关系数
    rank(x)           = 截面百分位排名
    """
    rank_open = open.rank(axis = 1, pct = True)
    rank_volume = volume.rank(axis=1  , pct = True)
    # 用 rolling corr：对每只股票，rank_open和rank_volume在时间轴上的10期相关
    alpha = -1 * rank_open.rolling(10).corr(rank_volume)
    return alpha

def alpha004(low):  #双显著
    """
    Alpha04 = -1 * Ts_Rank(rank(low), 9)
    low_df: 日频最低价 pivot表 (trade_date × ts_code)
    """
    rank_low = low.rank(axis = 1, pct = True)
    #对滚动9天的low进行排序，取第9天的排序
    ts_rank_low = rank_low.rolling(9).apply(lambda x:pd.Series(x).rank().iloc[-1]/len(x), raw = True)
    alpha = -1 * ts_rank_low
    return alpha

def alpha005(open, vwap, close):   #双显著
    """
    Alpha05 = rank(open - mean(vwap, 10)) * (-1 * abs(rank(close - vwap)))
    """
    vwap_ma = vwap.rolling(10).mean()
    rank_p1 = (open - vwap_ma).rank(axis = 1 , pct = True)
    rank_p2 = -1*abs((close - vwap).rank(axis = 1, pct = True))
    alpha = (rank_p1 * rank_p2).rank(axis = 1, pct = True)
    return alpha

def alpha006(open, volume): #双显著
    """
    Alpha#06: (-1*correlation(open, volume, 10))
    """
    alpha = (-1) * open.rolling(10).corr(volume)
    return alpha

def alpha007(close, volume):
    """
    Alpha#07: (adv20 < volume) ? (-1 * ts_rank(abs(delta(close,7)), 60) * sign(delta(close,7))) : -1
    """
    adv20 = volume.rolling(20).mean()
    delta7 = close.diff(7)
    # ts_rank：过去60日窗口内的升序百分比排名
    ts_rank = delta7.abs().rolling(60).rank(pct = True)
    # 放量分支
    active_signal = -1 * ts_rank * np.sign(delta7)
    # 条件合并
    alpha = pd.DataFrame(
        np.where(volume > adv20, active_signal, -1),
        index=close.index,
        columns=close.columns
    )
    return alpha

def alpha008(open, close):
    """
    Alpha#08:  =  (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
    """
    ret = close.pct_change()
    product = (open.rolling(5).sum()) * (ret.rolling(5).sum())
    diff = product - product.shift(10)
    alpha = (-1) * diff.rank(axis = 1, pct = True)
    return alpha

def alpha009(close):
    """
    Alpha#09: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    """
    ts_min = close.diff().rolling(5).min()
    ts_max = close.diff().rolling(5).max()
    condition = (ts_min > 0) | (ts_max < 0)
    alpha = np.where(condition, close.diff(), -close.diff())
    return pd.DataFrame(alpha, index=close.index, columns=close.columns)

def alpha010(close):
    """
    Alpha#10:  rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
    """
    ts_min = close.diff().rolling(4).min()
    ts_max = close.diff().rolling(4).max()
    condition = (ts_min> 0) | (ts_max< 0)
    direction = pd.DataFrame(np.where(condition, close.diff(), -close.diff()),index = close.index , columns = close.columns)
    alpha = direction.rank(axis = 1, pct = True)
    return alpha

def alpha011(close, vwap, volume):
    """
    Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    """
    ts_max = (vwap - close).rolling(3).max().rank(axis = 1,pct = True)
    ts_min = (vwap - close).rolling(3).min().rank(axis = 1,pct = True)
    vol_del = volume.diff(3).rank(axis = 1,pct = True)
    alpha = (ts_max + ts_min) * vol_del
    return alpha

def alpha012(close, vol):
    """
     Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    """
    alpha = np.sign(vol.diff())*(-1 * close.diff())
    return alpha

def alpha013(close, vol):
    """
    Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))
    """
    rank_close = close.rank(axis=1, pct=True)
    rank_vol = vol.rank(axis=1, pct=True)
    roll_cov = rank_close.rolling(5).cov(rank_vol)
    alpha = -1 * roll_cov.rank(axis=1, pct=True)
    return alpha

def alpha014(close, open, vol):
    """
    Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    """
    ret = close.pct_change()
    rank = -1 * ret.diff(3).rank(axis=1, pct=True)
    corr = open.rolling(10).corr(vol)
    alpha = rank * corr
    return alpha

def alpha015(high, vol):
    """
    Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    """
    rank_high = high.rank(axis=1, pct=True)
    rank_vol = vol.rank(axis=1, pct=True)
    roll_corr = rank_high.rolling(3).corr(rank_vol)
    alpha = -1 * roll_corr.rank(axis=1, pct=True).rolling(3).sum()
    return alpha

def alpha016(high, vol):
    """
    Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))
    """
    rank_high = high.rank(axis=1, pct=True)
    rank_vol = vol.rank(axis=1, pct=True)
    roll_cov = rank_high.rolling(5).cov(rank_vol)
    alpha = -1 * roll_cov.rank(axis=1, pct=True)
    return alpha

def alpha017(close, vol):
    """
    Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
    """
    adv20 = vol.rolling(20).mean()

    ts_rank_close = close.rolling(10).rank(pct=True)
    delta2_close = close.diff().diff()
    ts_rank_vol = (vol / adv20).rolling(5).rank(pct=True)

    alpha = (
        (-1 * ts_rank_close.rank(axis=1, pct=True))
        * delta2_close.rank(axis=1, pct=True)
        * ts_rank_vol.rank(axis=1, pct=True)
    )
    return alpha

def alpha018(close, open):
    """
    Alpha#18: -1 * rank(ts_std(abs(close - open), 5) + (close - open) + ts_corr(close, open, 10))
    """
    part1 = (close - open).abs().rolling(5).std()
    part2 = close - open
    part3 = close.rolling(10).corr(open)
    alpha = -1 * (part1 + part2 + part3).rank(axis=1, pct=True)
    return alpha

def alpha019(close):
    """
    Alpha#19: -1 * sign((close - delay(close,7)) + delta(close,7)) * (1 + rank(1 + ts_sum(returns,250)))
    """
    returns = close.pct_change()
    d7 = close - close.shift(7)
    sig = np.sign(d7 + d7)
    ret_sum = returns.rolling(250).sum()
    rnk = (1 + ret_sum).rank(axis=1, pct=True)
    alpha = -1 * sig * (1 + rnk)
    return alpha

def alpha020(close, open, high, low):
    """
    Alpha#20: -1 * rank(open - delay(high,1)) * rank(open - delay(close,1)) * rank(open - delay(low,1))
    """
    r1 = (open - high.shift(1)).rank(axis=1, pct=True)
    r2 = (open - close.shift(1)).rank(axis=1, pct=True)
    r3 = (open - low.shift(1)).rank(axis=1, pct=True)
    alpha = -1 * r1 * r2 * r3
    return alpha

def alpha021(close, volume):
    """
    Alpha#21:
    if (ts_mean(close,8) + ts_std(close,8)) < ts_mean(close,2): -1
    elif ts_mean(close,2) < (ts_mean(close,8) - ts_std(close,8)): 1
    elif volume / mean(volume,20) >= 1: 1
    else: -1
    """
    mean8 = close.rolling(8).mean()
    std8 = close.rolling(8).std()
    mean2 = close.rolling(2).mean()
    vol_ratio = volume / volume.rolling(20).mean()

    alpha = pd.DataFrame(-1.0, index=close.index, columns=close.columns)
    cond2 = mean2 < (mean8 - std8)
    alpha[cond2] = 1.0
    cond3 = (~cond2) & (vol_ratio >= 1)
    alpha[cond3] = 1.0
    cond1 = (mean8 + std8) < mean2
    alpha[cond1] = -1.0
    return alpha

def alpha022(close, high, volume):
    """
    Alpha#22: -1 * delta(ts_corr(high, volume, 5), 5) * rank(ts_std(close, 20))
    """
    corr_hv = high.rolling(5).corr(volume)
    alpha = -1 * corr_hv.diff(5) * close.rolling(20).std().rank(axis=1, pct=True)
    return alpha

def alpha023(high):
    """
    Alpha#23: if ts_mean(high,20) < high: -1 * delta(high,2) else: 0
    """
    mean20 = high.rolling(window=20).mean()
    delta2 = high.diff(2)
    alpha = np.where(high > mean20,-1 * delta2,0)
    return pd.DataFrame(alpha, index=high.index, columns=high.columns)

def alpha024(close):
    """
    Alpha#24:
    if delta(ts_mean(close,100), 100) / delay(close,100) <= 0.05: -1 * delta(close,3)
    else: -1 * (close - ts_min(close,100))
    """
    cond = (close.rolling(100).mean().diff(100) / close.shift(100)) <= 0.05
    branch1 = -1 * close.diff(3)
    branch2 = -1 * (close - close.rolling(100).min())
    alpha = branch1.where(cond, branch2)
    return alpha

def alpha025(close, high, volume, vwap):
    """
    Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    """
    returns = close.pct_change()
    adv20 = volume.rolling(20).mean()
    alpha = ((-1 * returns) * adv20 * vwap * (high - close)).rank(axis=1, pct=True)
    return alpha

def alpha026(high, volume):
    """
    Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    """
    ts_rank_vol = volume.rolling(5).rank(pct=True)
    ts_rank_high = high.rolling(5).rank(pct=True)
    corr = ts_rank_vol.rolling(5).corr(ts_rank_high)
    alpha = -1 * corr.rolling(3).max()
    return alpha

def alpha027(volume, vwap):
    """
    Alpha#27: if rank(mean(corr(rank(volume), rank(vwap), 6), 2)) > 0.5: -1 else: 1
    """
    rank_vol = volume.rank(axis=1, pct=True)
    rank_vwap = vwap.rank(axis=1, pct=True)
    corr = rank_vol.rolling(6).corr(rank_vwap)
    mean2 = corr.rolling(2).mean()
    rnk = mean2.rank(axis=1, pct=True)
    alpha = pd.DataFrame(np.where(rnk > 0.5, -1, 1),
                         index=volume.index, columns=volume.columns)
    return alpha

def alpha028(close, high, low, volume):
    """
    Alpha#28: scale(correlation(adv20, low, 5) + ((high + low) / 2) - close)
    """
    adv20 = volume.rolling(20).mean()
    corr = adv20.rolling(5).corr(low)
    raw = corr + (high + low) / 2 - close
    alpha = raw.div(raw.abs().sum(axis=1), axis=0)
    return alpha

def alpha029(close):
    """
    Alpha#29: ts_min(rank(rank(scale(log(ts_min(rank(rank(-1*rank(delta(close-1,5)))),2))))),5)
              + ts_rank(delay(-1*returns,6), 5)
    """

    returns = close.pct_change()
    delta = close.diff(5)
    r_delta = delta.rank(axis=1, pct=True)
    # rank(rank(-1 * rank(delta)))
    r1 = (-1 * r_delta).rank(axis=1, pct=True)
    r2 = r1.rank(axis=1, pct=True)
    # ts_min(..., 2)
    ts_min2 = r2.rolling(2).min()
    # log
    log_val = np.log(ts_min2 + 1e-10)
    # scale
    denom = log_val.abs().sum(axis=1)
    scaled = log_val.div(denom, axis=0)
    # rank(rank(scale(...)))
    r3 = scaled.rank(axis=1, pct=True)
    r4 = r3.rank(axis=1, pct=True)
    # ts_min(..., 5)
    part1 = r4.rolling(5).min()

    # ts_rank(delay(-returns, 6), 5)
    delayed = (-returns).shift(6)

    part2 = delayed.rolling(5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )
    alpha = part1 + part2
    return alpha

def alpha030(close, volume):
    """
    Alpha#30: (1 - rank((sign(close-delay(close,1)) + sign(delay(close,1)-delay(close,2))
               + sign(delay(close,2)-delay(close,3))))) * sum(volume,5) / sum(volume,20)
    """
    s1 = np.sign(close - close.shift(1))
    s2 = np.sign(close.shift(1) - close.shift(2))
    s3 = np.sign(close.shift(2) - close.shift(3))
    rnk = (s1 + s2 + s3).rank(axis=1, pct=True)
    alpha = (1 - rnk) * volume.rolling(5).sum() / volume.rolling(20).sum()
    return alpha

def alpha031(close, low, volume):
    """
    Alpha#31: rank(rank(rank(decay_linear(-1*rank(rank(delta(close,10))),10))))
              + rank(-1*delta(close,3)) + sign(scale(correlation(adv20, low, 12)))
    """
    adv20 = volume.rolling(20).mean()
    weights = np.arange(1, 11)
    decay_input = -1 * close.diff(10).rank(axis=1, pct=True).rank(axis=1, pct=True)
    decay = decay_input.rolling(10).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    r1 = decay.rank(axis=1, pct=True).rank(axis=1, pct=True).rank(axis=1, pct=True)
    r2 = (-1 * close.diff(3)).rank(axis=1, pct=True)
    corr = adv20.rolling(12).corr(low)
    scaled = corr.div(corr.abs().sum(axis=1), axis=0)
    alpha = r1 + r2 + np.sign(scaled)
    return alpha

def alpha032(close, vwap):
    """
    Alpha#32: scale((sum(close,7)/7 - close)) + 20*scale(correlation(vwap, delay(close,5), 230))
    """
    part1 = close.rolling(7).mean() - close
    scaled1 = part1.div(part1.abs().sum(axis=1), axis=0)
    corr = vwap.rolling(230).corr(close.shift(5))
    scaled2 = corr.div(corr.abs().sum(axis=1), axis=0)
    alpha = scaled1 + 20 * scaled2
    return alpha

def alpha033(close, open):
    """
    Alpha#33: rank(-1 * (1 - (open/close))^1)
    """
    alpha = (-1 * (1 - open / close)).rank(axis=1, pct=True)
    return alpha

def alpha034(close):
    """
    Alpha#34: rank((1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,1))))
    """
    returns = close.pct_change()
    std2 = returns.rolling(2).std()
    std5 = returns.rolling(5).std()
    part1 = 1 - (std2 / std5).rank(axis=1, pct=True)
    part2 = 1 - close.diff(1).rank(axis=1, pct=True)
    alpha = (part1 + part2).rank(axis=1, pct=True)
    return alpha

def alpha035(close, high, low, volume):
    """
    Alpha#35: ts_rank(volume,32) * (1 - ts_rank((close+high-low),16)) * (1 - ts_rank(returns,32))
    """
    returns = close.pct_change()
    ts_rank_vol = volume.rolling(32).rank(pct=True)
    ts_rank_chl = (close + high - low).rolling(16).rank(pct=True)
    ts_rank_ret = returns.rolling(32).rank(pct=True)
    alpha = ts_rank_vol * (1 - ts_rank_chl) * (1 - ts_rank_ret)
    return alpha

def alpha036(close, open, volume,vwap):
    """
    Alpha#36: 2.21*rank(corr(close-open, delay(volume,1),15)) + 0.7*rank(open-close)
              + 0.73*rank(ts_rank(delay(-returns,6),5)) + rank(abs(corr(vwap,adv20,6)))
              + 0.6*rank((mean(close,200)-open)*(close-open))
    """
    returns = close.pct_change()
    adv20 = volume.rolling(20).mean()

    part1 = 2.21 * (close - open).rolling(15).corr(volume.shift(1)).rank(axis=1, pct=True)
    part2 = 0.7 * (open - close).rank(axis=1, pct=True)
    part3 = 0.73 * (-returns).shift(6).rolling(5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False).rank(axis=1, pct=True)
    part4 = vwap.rolling(6).corr(adv20).abs().rank(axis=1, pct=True)
    part5 = 0.6 * ((close.rolling(200).mean() - open) * (close - open)).rank(axis=1, pct=True)

    alpha = part1 + part2 + part3 + part4 + part5
    return alpha

def alpha037(close, open):
    """
    Alpha#37: rank(corr(delay(open-close,1), close, 200)) + rank(open-close)
    """
    part1 = (open - close).shift(1).rolling(200).corr(close).rank(axis=1, pct=True)
    part2 = (open - close).rank(axis=1, pct=True)
    alpha = part1 + part2
    return alpha

def alpha038(close, open):
    """
    Alpha#38: (-1 * ts_rank(close,10)) * rank(close/open)
    """
    ts_rank_close = close.rolling(10).rank(pct=True)
    alpha = -1 * ts_rank_close * (close / open).rank(axis=1, pct=True)
    return alpha

def alpha039(close, volume):
    """
    Alpha#39: (-1 * rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20, 9)))))
              * (1 + rank(sum(returns,250)))
    """
    returns = close.pct_change()
    adv20 = volume.rolling(20).mean()
    weights = np.arange(1, 10, dtype=float)
    vol_ratio = volume / adv20
    decay = vol_ratio.rolling(9).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    part1 = -1 * (close.diff(7) * (1 - decay.rank(axis=1, pct=True))).rank(axis=1, pct=True)
    part2 = 1 + returns.rolling(250).sum().rank(axis=1, pct=True)
    alpha = part1 * part2
    return alpha

def alpha040(high, volume):
    """
    Alpha#40: (-1 * rank(stddev(high,10))) * corr(high, volume, 10)
    """
    part1 = -1 * high.rolling(10).std().rank(axis=1, pct=True)
    part2 = high.rolling(10).corr(volume)
    alpha = part1 * part2
    return alpha

def alpha041(high, low, vwap):
    """
    Alpha#41: (high * low)^0.5 - vwap
    """
    alpha = (high * low) ** 0.5 - vwap
    return alpha

def alpha042(close, vwap):
    """
    Alpha#42: rank(vwap - close) / rank(vwap + close)
    """
    alpha = (vwap - close).rank(axis=1, pct=True) / (vwap + close).rank(axis=1, pct=True)
    return alpha

def alpha043(close, volume):
    """
    Alpha#43: ts_rank(volume/adv20, 20) * ts_rank(-1*delta(close,7), 8)
    """
    adv20 = volume.rolling(20).mean()
    ts_rank_vol = (volume / adv20).rolling(20).rank(pct=True)
    ts_rank_delta = (-1 * close.diff(7)).rolling(8).rank(pct=True)
    alpha = ts_rank_vol * ts_rank_delta
    return alpha

def alpha044(high, volume):
    """
    Alpha#44: -1 * correlation(high, rank(volume), 5)
    """
    rank_volume = volume.rank(axis=1, pct=True)
    alpha = -high.rolling(5).corr(rank_volume)
    return alpha

def alpha045(close, volume):
    """
    Alpha#45: -1 * (rank(sum(delay(close,5),20)/20) * correlation(close,volume,2))* rank(correlation(sum(close,5), sum(close,20), 2))
    """
    rank_mean_delay = close.shift(5).rolling(20).mean().rank(axis=1, pct=True)
    corr_cv         = close.rolling(2).corr(volume)
    sum_close5      = close.rolling(5).sum()
    sum_close20     = close.rolling(20).sum()
    rank_corr       = sum_close5.rolling(2).corr(sum_close20).rank(axis=1, pct=True)
    alpha = -1 * (rank_mean_delay * corr_cv) * rank_corr
    return alpha

def alpha046(close):
    """
    Alpha#46:
      mid = ((delay(close,20)-delay(close,10))/10) - ((delay(close,10)-close)/10)
      mid > 0.25  -> -1
      mid < 0     ->  1
      else        -> -1 * (close - delay(close,1))
    """
    d10 = close.shift(10)
    d20 = close.shift(20)
    mid = (d20 - d10) / 10.0 - (d10 - close) / 10.0

    alpha = -close.diff(1)                         # else
    alpha = alpha.where(mid >= 0,    other=1.0)    # mid < 0    -> 1
    alpha = alpha.where(mid <= 0.25, other=-1.0)   # mid > 0.25 -> -1
    return alpha

def alpha047(close, high, vwap, volume):
    """
    Alpha#47: ((rank(1/close)*volume/adv20) * (high*rank(high-close))/(sum(high,5)/5)) - rank(vwap - delay(vwap,5))
    """
    adv20  = volume.rolling(20).mean()
    part1 = ((1.0 / close).rank(axis=1, pct=True) * volume / adv20)* (high * (high - close).rank(axis=1, pct=True))/ (high.rolling(5).sum() / 5.0)
    part2 = (vwap - vwap.shift(5)).rank(axis=1, pct=True)
    alpha = part1 - part2
    return alpha

def alpha048(close,industry):
    """
    Alpha#48:
      indneutralize(corr(delta(close,1), delta(delay(close,1),1), 250) * delta(close,1) / close,IndClass.subindustry) / sum((delta(close,1)/delay(close,1))^2, 250)
    """
    industry_s = industry['industry']   # Series: ts_code → 行业标签

    delta1      = close.diff(1)            # delta(close, 1)
    delay1      = close.shift(1)           # delay(close, 1)
    delta_delay = delay1.diff(1)           # delta(delay(close,1), 1)

    rolling_corr = delta1.rolling(250).corr(delta_delay)   # 250日滚动相关
    raw          = rolling_corr * delta1 / close            # 分子原始值

    numerator    = indneutralize(raw, industry_s) # 行业中性化

    daily_ret    = delta1 / delay1
    denominator  = (daily_ret ** 2).rolling(250).sum()
    alpha = numerator / denominator.replace(0, np.nan)
    return alpha

def alpha049(close):
    """
    Alpha#49:
      mid = ((delay(close,20)-delay(close,10))/10) - ((delay(close,10)-close)/10)
      mid < -0.1  -> 1
      else        -> -1 * (close - delay(close,1))
    """
    d10 = close.shift(10)
    d20 = close.shift(20)
    mid = (d20 - d10) / 10.0 - (d10 - close) / 10.0

    alpha = -close.diff(1)                        # else
    alpha = alpha.where(mid >= -0.1, other=1.0)   # mid < -0.1 -> 1
    return alpha

def alpha050(volume, vwap):
    """
    Alpha#50: -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)
    """
    rank_volume = volume.rank(axis=1, pct=True)
    rank_vwap   = vwap.rank(axis=1, pct=True)
    corr        = rank_volume.rolling(5).corr(rank_vwap)
    rank_corr   = corr.rank(axis=1, pct=True)
    alpha       = -rank_corr.rolling(5).max()
    return alpha

def alpha051(close):
    """
    Alpha#51:
      mid = ((delay(close,20)-delay(close,10))/10) - ((delay(close,10)-close)/10)
      mid < -0.05 -> 1
      else        -> -1 * (close - delay(close,1))
    """
    d10 = close.shift(10)
    d20 = close.shift(20)
    mid = (d20 - d10) / 10.0 - (d10 - close) / 10.0

    alpha = -close.diff(1)                         # else
    alpha = alpha.where(mid >= -0.05, other=1.0)   # mid < -0.05 -> 1
    return alpha

def alpha052(low, volume, close):
    """
    Alpha#52: ((-ts_min(low,5) + delay(ts_min(low,5),5))
               * rank((sum(returns,240) - sum(returns,20)) / 220))
               * ts_rank(volume, 5)
    """
    returns    = close.pct_change()
    low5_min   = low.rolling(5).min()
    delta_min  = -low5_min + low5_min.shift(5)
    rank_ret   = (
        (returns.rolling(240).sum() - returns.rolling(20).sum()) / 220
    ).rank(axis=1, pct=True)
    ts_rank_vol = volume.rolling(5).rank(pct=True)
    alpha = delta_min * rank_ret * ts_rank_vol
    return alpha

def alpha053(close, low, high):
    """
    Alpha#53: -1 * delta(((close-low)-(high-close)) / (close-low), 9)
    """
    inner = ((close - low) - (high - close)) / (close - low)
    alpha = -inner.diff(9)
    return alpha

def alpha054(close, low, high, open):
    """
    Alpha#54: (-1 * (low - close) * (open^5)) / ((low - high) * (close^5))
    """
    alpha = (-1 * (low - close) * (open ** 5)) / ((low - high) * (close ** 5))
    return alpha

def alpha055(close, low, high, volume):
    """
    Alpha#55: -1 * correlation(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))),
                               rank(volume), 6)
    """
    low12  = low.rolling(12).min()
    high12 = high.rolling(12).max()
    rank_price = ((close - low12) / (high12 - low12)).rank(axis=1, pct=True)
    rank_vol   = volume.rank(axis=1, pct=True)
    alpha = -rank_price.rolling(6).corr(rank_vol)
    return alpha

def alpha056(close, cap):
    """
    Alpha#56: 0 - 1 * (rank(sum(returns,10)/sum(sum(returns,2),3)) * rank(returns*cap))
    cap: 月末index的DataFrame，函数内reindex到交易日并ffill
    """
    returns = close.pct_change()

    # 将月末cap对齐到交易日
    cap_daily = cap.reindex(close.index, method="ffill")

    sum_ret2 = returns.rolling(2).sum()
    rank_ratio = (returns.rolling(10).sum() / sum_ret2.rolling(3).sum()).rank(axis=1, pct=True)
    rank_ret_cap = (returns * cap_daily).rank(axis=1, pct=True)
    alpha = -(rank_ratio * rank_ret_cap)
    return alpha

def alpha057(close, vwap):
    """
    Alpha#57: 0 - 1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))
    """
    ts_argmax   = close.rolling(30).apply(lambda x: int(np.argmax(x)), raw=True)
    rank_argmax = ts_argmax.rank(axis=1, pct=True)

    # decay_linear window=2: weights [1,2]/3
    w = np.array([1.0, 2.0])
    w = w / w.sum()
    decay = rank_argmax.rolling(2).apply(lambda x: (x * w).sum(), raw=True)

    alpha = -((close - vwap) / decay)
    return alpha

def alpha058(vwap, volume, industry):
    """
    Alpha#58: -1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, sector), volume, 4), 8), 6)
    industry_s: index=ts_code, values=行业标签(sector级别)
    """
    industry_s = industry['industry']   # Series: ts_code → 行业标签
    vwap_neu = indneutralize(vwap, industry_s)

    corr = vwap_neu.rolling(4).corr(volume)

    # decay_linear window=8: weights [1..8]/36
    w = np.arange(1, 9, dtype=float)
    w = w / w.sum()
    decay = corr.rolling(8).apply(lambda x: (x * w).sum(), raw=True)

    alpha = -decay.rolling(6).rank(pct=True)
    return alpha

def alpha059(vwap, volume, industry):
    """
    Alpha#59: -1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap*0.728317 + vwap*(1-0.728317), industry), volume, 4), 16), 8)
    注意: vwap*0.728317 + vwap*(1-0.728317) = vwap，系数化简后就是vwap本身
    industry_s: index=ts_code, values=行业标签(industry级别)
    """
    # vwap * 0.728317 + vwap * (1 - 0.728317) = vwap
    industry_s = industry['industry']   # Series: ts_code → 行业标签
    vwap_neu = indneutralize(vwap, industry_s)
    corr = vwap_neu.rolling(4).corr(volume)
    # decay_linear window=16: weights [1..16]/136
    w = np.arange(1, 17, dtype=float)
    w = w / w.sum()
    decay = corr.rolling(16).apply(lambda x: (x * w).sum(), raw=True)

    alpha = -decay.rolling(8).rank(pct=True)
    return alpha

def alpha060(close, low, high, volume):
    """
    Alpha#60: 0 - 1 * (2*scale(rank(((close-low)-(high-close))/(high-low)*volume))
                        - scale(rank(ts_argmax(close, 10))))
    """
    inner       = ((close - low) - (high - close)) / (high - low) * volume
    rank_inner  = inner.rank(axis=1, pct=True)
    ts_argmax   = close.rolling(10).apply(lambda x: int(np.argmax(x)), raw=True)
    rank_argmax = ts_argmax.rank(axis=1, pct=True)

    scale_inner  = rank_inner.div(rank_inner.abs().sum(axis=1), axis=0)
    scale_argmax = rank_argmax.div(rank_argmax.abs().sum(axis=1), axis=0)

    alpha = -(2 * scale_inner - scale_argmax)
    return alpha

def alpha061(vwap, volume):
    """
    Alpha#61: rank(vwap - ts_min(vwap, 16)) < rank(correlation(vwap, adv180, 18))
    最后结果改为差值，不是bool
    """
    adv180     = volume.rolling(180).mean()
    rank_left  = (vwap - vwap.rolling(16).min()).rank(axis=1, pct=True)
    rank_right = vwap.rolling(18).corr(adv180).rank(axis=1, pct=True)
    alpha = rank_right - rank_left
    return alpha

def alpha062(vwap, open, high, low, volume):
    """
    Alpha#62: rank(correlation(vwap, sum(adv20,22), 10))
              < rank((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high)))
              最后结果改为差值，不是bool
    """
    adv20     = volume.rolling(20).mean()
    sum_adv20 = adv20.rolling(22).sum()
    rank_corr = vwap.rolling(10).corr(sum_adv20).rank(axis=1, pct=True)

    rank_open = open.rank(axis=1, pct=True)
    rank_hl2  = ((high + low) / 2).rank(axis=1, pct=True)
    rank_high = high.rank(axis=1, pct=True)
    rank_right = (rank_open + rank_open - rank_hl2 - rank_high).rank(axis=1, pct=True)

    alpha = -(rank_right - rank_corr)
    return alpha

def alpha063(close, vwap, open, volume, industry):
    """
    Alpha#63: (rank(decay_linear(delta(IndNeutralize(close, industry), 2), 8))
               - rank(decay_linear(correlation(vwap*0.318108 + open*0.681892, sum(adv180,37), 14), 12))) * -1
    """
    industry_s = industry['industry']
    close_neu  = indneutralize(close, industry_s)

    adv180     = volume.rolling(180).mean()
    sum_adv180 = adv180.rolling(37).sum()
    price      = vwap * 0.318108 + open * (1 - 0.318108)

    w8  = np.arange(1, 9,  dtype=float); w8  /= w8.sum()
    w12 = np.arange(1, 13, dtype=float); w12 /= w12.sum()

    decay1 = close_neu.diff(2).rolling(8).apply(lambda x: (x * w8).sum(),   raw=True)
    decay2 = price.rolling(14).corr(sum_adv180).rolling(12).apply(lambda x: (x * w12).sum(), raw=True)

    alpha = -(decay1.rank(axis=1, pct=True) - decay2.rank(axis=1, pct=True))
    return alpha

def alpha064(open, low, high, vwap, volume):
    """
    Alpha#64: (rank(correlation(sum(open*0.178404 + low*0.821596, 13), sum(adv120,13), 17))
               < rank(delta(high*0.178404/2 + low*0.178404/2 + vwap*0.821596, 4))) * -1
               最后结果改为差值，不是bool
    """
    adv120    = volume.rolling(120).mean()
    price1    = open * 0.178404 + low * (1 - 0.178404)
    price2    = ((high + low) / 2) * 0.178404 + vwap * (1 - 0.178404)

    rank_corr  = price1.rolling(13).sum().rolling(17).corr(adv120.rolling(13).sum()).rank(axis=1, pct=True)
    rank_delta = price2.diff(4).rank(axis=1, pct=True)

    alpha = -(rank_delta - rank_corr)
    return alpha

def alpha065(open, vwap, volume):
    """
    Alpha#65: (rank(correlation(open*0.00817205 + vwap*0.99182795, sum(adv60,9), 6))
               < rank(open - ts_min(open,14))) * -1
    """
    adv60     = volume.rolling(60).mean()
    price     = open * 0.00817205 + vwap * (1 - 0.00817205)

    rank_corr  = price.rolling(6).corr(adv60.rolling(9).sum()).rank(axis=1, pct=True)
    rank_delta = (open - open.rolling(14).min()).rank(axis=1, pct=True)

    alpha = -(rank_delta - rank_corr)
    return alpha

def alpha066(low, vwap, open, high):
    """
    Alpha#66: (rank(decay_linear(delta(vwap,4), 7))
               + Ts_Rank(decay_linear((low - vwap)/(open-(high+low)/2), 11), 7)) * -1
    low*0.96633 + low*(1-0.96633) = low
    """
    w7  = np.arange(1, 8,  dtype=float); w7  /= w7.sum()
    w11 = np.arange(1, 12, dtype=float); w11 /= w11.sum()

    decay1 = vwap.diff(4).rolling(7).apply(lambda x: (x * w7).sum(), raw=True)
    inner  = (low - vwap) / (open - (high + low) / 2)
    decay2 = inner.rolling(11).apply(lambda x: (x * w11).sum(), raw=True)

    alpha = -(decay1.rank(axis=1, pct=True) + decay2.rolling(7).rank(pct=True))
    return alpha

def alpha067(high, vwap, volume, industry):
    """
    Alpha#67: (rank(high - ts_min(high,2)) ^ rank(correlation(IndNeutralize(vwap, sector),
               IndNeutralize(adv20, subindustry), 6))) * -1
    用同一个industry_s做两次中性化
    """
    industry_s = industry['industry']
    adv20      = volume.rolling(20).mean()

    vwap_neu  = indneutralize(vwap,  industry_s)
    adv20_neu = indneutralize(adv20, industry_s)

    rank_left  = (high - high.rolling(2).min()).rank(axis=1, pct=True)
    rank_right = vwap_neu.rolling(6).corr(adv20_neu).rank(axis=1, pct=True)

    alpha = -(rank_left ** rank_right)
    return alpha

def alpha068(high, close, low, volume):
    """
    Alpha#68: (Ts_Rank(correlation(rank(high), rank(adv15), 9), 14)
               < rank(delta(close*0.518371 + low*0.481629, 1))) * -1
    """
    adv15     = volume.rolling(15).mean()
    price     = close * 0.518371 + low * (1 - 0.518371)

    rank_left  = (high.rank(axis = 1, pct = True)).rolling(9).corr(adv15.rank(axis = 1, pct = True)).rolling(14).rank(pct=True)
    rank_right = price.diff(1).rank(axis=1, pct=True)

    alpha = -(rank_right - rank_left)
    return alpha

def alpha069(close, vwap, volume, industry):
    """
    Alpha#69: (rank(ts_max(delta(IndNeutralize(vwap, industry), 3), 5))
               ^ Ts_Rank(correlation(close*0.490655 + vwap*0.509345, adv20, 5), 9)) * -1
    """
    industry_s = industry['industry']
    adv20      = volume.rolling(20).mean()
    vwap_neu   = indneutralize(vwap, industry_s)
    price      = close * 0.490655 + vwap * (1 - 0.490655)

    rank_left  = vwap_neu.diff(3).rolling(5).max().rank(axis=1, pct=True)
    rank_right = price.rolling(5).corr(adv20).rolling(9).rank(pct=True)

    alpha = -(rank_left ** rank_right)
    return alpha

def alpha070(close, vwap, volume, industry):
    """
    Alpha#70: ((rank(delta(vwap, 1.29456))
                ^ Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
    """
    industry_s = industry['industry']
    adv50      = volume.rolling(50).mean()
    close_neu  = indneutralize(close, industry_s)

    rank_left     = vwap.diff(1).rank(axis=1, pct=True)
    ts_rank_right = close_neu.rolling(18).corr(adv50).rolling(18).rank(pct=True)
    alpha = -(rank_left ** ts_rank_right)
    return alpha

def alpha071(close, low, open, vwap, volume):
    """
    Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close,3), Ts_Rank(adv180,12), 18), 4), 16),
                  Ts_Rank(decay_linear(rank((low+open-2*vwap)^2), 16), 4))
    """
    adv180 = volume.rolling(180).mean()

    w4  = np.arange(1, 5,  dtype=float); w4  /= w4.sum()
    w16 = np.arange(1, 17, dtype=float); w16 /= w16.sum()

    ts_rank_close  = close.rolling(3).rank(pct=True)
    ts_rank_adv180 = adv180.rolling(12).rank(pct=True)

    part1 = (ts_rank_close.rolling(18).corr(ts_rank_adv180).rolling(4).apply(lambda x: (x * w4).sum(), raw=True).rolling(16).rank(pct=True))
    inner = ((low + open - 2 * vwap) ** 2).rank(axis=1, pct=True)
    part2 = (inner.rolling(16).apply(lambda x: (x * w16).sum(), raw=True).rolling(4).rank(pct=True))
    alpha = pd.DataFrame(np.maximum(part1.values, part2.values),index=close.index, columns=close.columns)
    return alpha

def alpha072(high, low, vwap, volume):
    """
    Alpha#72: rank(decay_linear(correlation((high+low)/2, adv40, 9), 10))
              / rank(decay_linear(correlation(Ts_Rank(vwap,4), Ts_Rank(volume,19), 7), 3))
    """
    adv40 = volume.rolling(40).mean()

    w10 = np.arange(1, 11, dtype=float); w10 /= w10.sum()
    w3  = np.arange(1, 4,  dtype=float); w3  /= w3.sum()

    hl2    = (high + low) / 2
    decay1 = (hl2.rolling(9).corr(adv40).rolling(10).apply(lambda x: (x * w10).sum(), raw=True))

    ts_rank_vwap   = vwap.rolling(4).rank(pct=True)
    ts_rank_volume = volume.rolling(19).rank(pct=True)
    decay2 = (ts_rank_vwap.rolling(7).corr(ts_rank_volume).rolling(3).apply(lambda x: (x * w3).sum(), raw=True))

    alpha = decay1.rank(axis=1, pct=True) / decay2.rank(axis=1, pct=True)
    return alpha

def alpha073(open, low, vwap):
    """
    Alpha#73: max(rank(decay_linear(delta(vwap,5), 3)),
                  Ts_Rank(decay_linear((-1*(delta(open*0.147155+low*0.852845,2)/(open*0.147155+low*0.852845))), 3), 17)) * -1
    """
    w3 = np.arange(1, 4, dtype=float); w3 /= w3.sum()

    decay1 = (vwap.diff(5).rolling(3).apply(lambda x: (x * w3).sum(), raw=True))

    price  = open * 0.147155 + low * (1 - 0.147155)
    inner  = -1 * price.diff(2) / price
    decay2 = inner.rolling(3).apply(lambda x: (x * w3).sum(), raw=True)

    part1 = decay1.rank(axis=1, pct=True)
    part2 = decay2.rolling(17).rank(pct=True)

    alpha = -pd.DataFrame(np.maximum(part1.values, part2.values),index=open.index, columns=open.columns)
    return alpha

def alpha074(close, high, vwap, volume):
    """
    Alpha#74: (rank(correlation(close, sum(adv30,37), 15))
               < rank(correlation(rank(high*0.0261661 + vwap*0.9738339), rank(volume), 11))) * -1
    """
    adv30     = volume.rolling(30).mean()
    price     = high * 0.0261661 + vwap * (1 - 0.0261661)

    rank_left  = close.rolling(15).corr(adv30.rolling(37).sum()).rank(axis=1, pct=True)
    rank_right = (price.rank(axis=1, pct=True).rolling(11).corr(volume.rank(axis=1, pct=True)).rank(axis=1, pct=True))
    alpha = -(rank_right - rank_left)
    return alpha

def alpha075(vwap, low, volume):
    """
    Alpha#75: rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv50), 12))
    """
    adv50      = volume.rolling(50).mean()
    rank_left  = vwap.rolling(4).corr(volume).rank(axis=1, pct=True)
    rank_right = (low.rank(axis=1, pct=True).rolling(12).corr(adv50.rank(axis=1, pct=True)).rank(axis=1, pct=True))
    alpha = rank_right - rank_left
    return alpha

def alpha076(low, vwap, volume, industry):
    """
    Alpha#76: max(rank(decay_linear(delta(vwap,1), 12)),
                  Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low,sector), adv81,8), 20), 17), 19)) * -1
    """
    industry_s = industry['industry']
    adv81      = volume.rolling(81).mean()
    low_neu    = indneutralize(low, industry_s)

    w12 = np.arange(1, 13, dtype=float); w12 /= w12.sum()
    w17 = np.arange(1, 18, dtype=float); w17 /= w17.sum()

    decay1 = (vwap.diff(1).rolling(12).apply(lambda x: (x * w12).sum(), raw=True)).rank(axis=1, pct=True)

    inner  = low_neu.rolling(8).corr(adv81).rolling(20).rank(pct=True)
    decay2 = (inner.rolling(17).apply(lambda x: (x * w17).sum(), raw=True).rolling(19).rank(pct=True))

    alpha = -pd.DataFrame(np.maximum(decay1.values, decay2.values),index=low.index, columns=low.columns)
    return alpha

def alpha077(high, low, vwap, volume):
    """
    Alpha#77: min(rank(decay_linear(((high+low)/2+high) - (vwap+high), 20)),
                  rank(decay_linear(correlation((high+low)/2, adv40, 3), 6)))
    (hl2+high) - (vwap+high) = hl2 - vwap
    """
    adv40 = volume.rolling(40).mean()
    hl2   = (high + low) / 2

    w20 = np.arange(1, 21, dtype=float); w20 /= w20.sum()
    w6  = np.arange(1, 7,  dtype=float); w6  /= w6.sum()

    decay1 = ((hl2 - vwap).rolling(20).apply(lambda x: (x * w20).sum(), raw=True)).rank(axis=1, pct=True)
    decay2 = (hl2.rolling(3).corr(adv40).rolling(6).apply(lambda x: (x * w6).sum(), raw=True)).rank(axis=1, pct=True)

    alpha = pd.DataFrame(np.minimum(decay1.values, decay2.values),index=high.index, columns=high.columns)
    return alpha

def alpha078(low, vwap, volume):
    """
    Alpha#78: rank(correlation(sum(low*0.352233+vwap*0.647767, 20), sum(adv40,20), 7))
              ^ rank(correlation(rank(vwap), rank(volume), 6))
    """
    adv40  = volume.rolling(40).mean()
    price  = low * 0.352233 + vwap * (1 - 0.352233)

    rank_left  = (price.rolling(20).sum().rolling(7).corr(adv40.rolling(20).sum()).rank(axis=1, pct=True))
    rank_right = (vwap.rank(axis=1, pct=True).rolling(6).corr(volume.rank(axis=1, pct=True)).rank(axis=1, pct=True))

    alpha = rank_left ** rank_right
    return alpha

def alpha079(close, open, vwap, volume, industry):
    """
    Alpha#79: rank(delta(IndNeutralize(close*0.60733+open*0.39267, sector), 1))
              < rank(correlation(Ts_Rank(vwap,4), Ts_Rank(adv150,9), 15))
    """
    industry_s = industry['industry']
    adv150     = volume.rolling(150).mean()
    price      = close * 0.60733 + open * (1 - 0.60733)
    price_neu  = indneutralize(price, industry_s)

    rank_left  = price_neu.diff(1).rank(axis=1, pct=True)
    rank_right = (vwap.rolling(4).rank(pct=True).rolling(15).corr(adv150.rolling(9).rank(pct=True)).rank(axis=1, pct=True))

    alpha = rank_right - rank_left
    return alpha

def alpha080(open, high, volume, industry):
    """
    Alpha#80: (rank(sign(delta(IndNeutralize(open*0.868128+high*0.131872, industry), 4)))
               ^ Ts_Rank(correlation(high, adv10, 5), 6)) * -1
    """
    industry_s = industry['industry']
    adv10      = volume.rolling(10).mean()
    price      = open * 0.868128 + high * (1 - 0.868128)
    price_neu  = indneutralize(price, industry_s)

    rank_left  = np.sign(price_neu.diff(4)).rank(axis=1, pct=True)
    rank_right = high.rolling(5).corr(adv10).rolling(6).rank(pct=True)

    alpha = -(rank_left ** rank_right)
    return alpha

def alpha081(vwap, volume):
    """
    Alpha#81: (rank(log(product(rank(rank(correlation(vwap, sum(adv10,49), 8))^4), 15)))
               < rank(correlation(rank(vwap), rank(volume), 5))) * -1
    """
    adv10      = volume.rolling(10).mean()
    corr       = vwap.rolling(8).corr(adv10.rolling(49).sum())
    inner      = (corr.rank(axis=1, pct=True) ** 4).rank(axis=1, pct=True)
    rank_left  = np.log(inner.rolling(15).apply(np.prod, raw=True)).rank(axis=1, pct=True)
    rank_right = (vwap.rank(axis=1, pct=True).rolling(5).corr(volume.rank(axis=1, pct=True)).rank(axis=1, pct=True))
    alpha = -(rank_right - rank_left)
    return alpha

def alpha082(open, volume, industry):
    """
    Alpha#82: min(rank(decay_linear(delta(open,1), 15)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume,sector), open, 17), 7), 13)) * -1
    open*0.634196 + open*(1-0.634196) = open
    """
    industry_s = industry['industry']
    vol_neu    = indneutralize(volume, industry_s)

    w15 = np.arange(1, 16, dtype=float); w15 /= w15.sum()
    w7  = np.arange(1, 8,  dtype=float); w7  /= w7.sum()

    part1 = (open.diff(1).rolling(15).apply(lambda x: (x * w15).sum(), raw=True).rank(axis=1, pct=True))
    part2 = (vol_neu.rolling(17).corr(open).rolling(7).apply(lambda x: (x * w7).sum(), raw=True).rolling(13).rank(pct=True))

    alpha = -pd.DataFrame(np.minimum(part1.values, part2.values),index=open.index, columns=open.columns)
    return alpha

def alpha083(high, low, close, vwap, volume):
    """
    Alpha#83: (rank(delay((high-low)/(sum(close,5)/5), 2)) * rank(rank(volume)))
              / (((high-low)/(sum(close,5)/5)) / (vwap - close))
    """
    hl_ratio  = (high - low) / close.rolling(5).mean()
    rank_left = hl_ratio.shift(2).rank(axis=1, pct=True)
    rank_vol  = volume.rank(axis=1, pct=True).rank(axis=1, pct=True)
    alpha     = (rank_left * rank_vol) / (hl_ratio / (vwap - close))
    return alpha

def alpha084(vwap, close):
    """
    Alpha#84: SignedPower(Ts_Rank(vwap - ts_max(vwap,15), 21), delta(close,5))
    SignedPower(x, e) = sign(x) * abs(x)^e
    """
    base  = (vwap - vwap.rolling(15).max()).rolling(21).rank(pct=True)
    exp   = close.diff(5)
    alpha = np.sign(base) * (base.abs() ** exp)
    return alpha

def alpha085(high, close, low, volume):
    """
    Alpha#85: rank(correlation(high*0.876703+close*0.123297, adv30, 10))
              ^ rank(correlation(Ts_Rank((high+low)/2, 4), Ts_Rank(volume,10), 7))
    """
    adv30      = volume.rolling(30).mean()
    price      = high * 0.876703 + close * (1 - 0.876703)
    rank_left  = price.rolling(10).corr(adv30).rank(axis=1, pct=True)
    rank_right = (((high + low) / 2).rolling(4).rank(pct=True).rolling(7).corr(volume.rolling(10).rank(pct=True)).rank(axis=1, pct=True))
    alpha = rank_left ** rank_right
    return alpha

def alpha086(close, vwap, volume):
    """
    Alpha#86: (Ts_Rank(correlation(close, sum(adv20,14), 6), 20)
               < rank((open+close)-(vwap+open))) * -1
    (open+close)-(vwap+open) = close - vwap
    """
    adv20      = volume.rolling(20).mean()
    rank_left  = close.rolling(6).corr(adv20.rolling(14).sum()).rolling(20).rank(pct=True)
    rank_right = (close - vwap).rank(axis=1, pct=True)
    alpha      = -(rank_right - rank_left)
    return alpha

def alpha087(close, vwap, volume, industry):
    """
    Alpha#87: max(rank(decay_linear(delta(close*0.369701+vwap*0.630299, 2), 3)),
                  Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,industry), close, 13)), 5), 14)) * -1
    """
    industry_s = industry['industry']
    adv81      = volume.rolling(81).mean()
    adv81_neu  = indneutralize(adv81, industry_s)
    price      = close * 0.369701 + vwap * (1 - 0.369701)

    w3 = np.arange(1, 4, dtype=float); w3 /= w3.sum()
    w5 = np.arange(1, 6, dtype=float); w5 /= w5.sum()

    part1 = (price.diff(2).rolling(3).apply(lambda x: (x * w3).sum(), raw=True).rank(axis=1, pct=True))
    part2 = (adv81_neu.rolling(13).corr(close).abs().rolling(5).apply(lambda x: (x * w5).sum(), raw=True).rolling(14).rank(pct=True))

    alpha = -pd.DataFrame(np.maximum(part1.values, part2.values),index=close.index, columns=close.columns)
    return alpha

def alpha088(open, low, high, close, volume):
    """
    Alpha#88: min(rank(decay_linear((rank(open)+rank(low))-(rank(high)+rank(close)), 8)),
                  Ts_Rank(decay_linear(correlation(Ts_Rank(close,8), Ts_Rank(adv60,21), 8), 7), 3))
    """
    adv60 = volume.rolling(60).mean()

    w8 = np.arange(1, 9, dtype=float); w8 /= w8.sum()
    w7 = np.arange(1, 8, dtype=float); w7 /= w7.sum()

    inner = (open.rank(axis=1, pct=True) + low.rank(axis=1, pct=True)
             - high.rank(axis=1, pct=True) - close.rank(axis=1, pct=True))
    part1 = (inner.rolling(8).apply(lambda x: (x * w8).sum(), raw=True).rank(axis=1, pct=True))
    part2 = (close.rolling(8).rank(pct=True)
             .rolling(8).corr(adv60.rolling(21).rank(pct=True))
             .rolling(7).apply(lambda x: (x * w7).sum(), raw=True)
             .rolling(3).rank(pct=True))

    alpha = pd.DataFrame(np.minimum(part1.values, part2.values),index=close.index, columns=close.columns)
    return alpha

def alpha089(low, vwap, volume, industry):
    """
    Alpha#89: Ts_Rank(decay_linear(correlation(low, adv10, 7), 6), 4)
              - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,industry), 3), 10), 15)
    注意: low*0.967285 + low*(1-0.967285) = low
    """
    industry_s = industry['industry']
    adv10      = volume.rolling(10).mean()
    vwap_neu   = indneutralize(vwap, industry_s)

    w6  = np.arange(1, 7,  dtype=float); w6  /= w6.sum()
    w10 = np.arange(1, 11, dtype=float); w10 /= w10.sum()

    part1 = (low.rolling(7).corr(adv10)
             .rolling(6).apply(lambda x: (x * w6).sum(), raw=True)
             .rolling(4).rank(pct=True))
    part2 = (vwap_neu.diff(3)
             .rolling(10).apply(lambda x: (x * w10).sum(), raw=True)
             .rolling(15).rank(pct=True))

    alpha = part1 - part2
    return alpha

def alpha090(close, low, volume, industry):
    """
    Alpha#90: (rank(close - ts_max(close,5))
               ^ Ts_Rank(correlation(IndNeutralize(adv40,subindustry), low, 5), 3)) * -1
    """
    industry_s = industry['industry']
    adv40      = volume.rolling(40).mean()
    adv40_neu  = indneutralize(adv40, industry_s)

    rank_left  = (close - close.rolling(5).max()).rank(axis=1, pct=True)
    rank_right = adv40_neu.rolling(5).corr(low).rolling(3).rank(pct=True)

    alpha = -(rank_left ** rank_right)
    return alpha

def alpha091(close, vwap, volume, industry):
    """
    Alpha#91: (Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,industry), volume, 10), 16), 4), 5)
               - rank(decay_linear(correlation(vwap, adv30, 4), 3))) * -1
    """
    industry_s = industry['industry']
    adv30      = volume.rolling(30).mean()
    close_neu  = indneutralize(close, industry_s)

    w16 = np.arange(1, 17, dtype=float); w16 /= w16.sum()
    w4  = np.arange(1, 5,  dtype=float); w4  /= w4.sum()
    w3  = np.arange(1, 4,  dtype=float); w3  /= w3.sum()

    part1 = (close_neu.rolling(10).corr(volume)
             .rolling(16).apply(lambda x: (x * w16).sum(), raw=True)
             .rolling(4).apply(lambda x: (x * w4).sum(), raw=True)
             .rolling(5).rank(pct=True))
    part2 = (vwap.rolling(4).corr(adv30)
             .rolling(3).apply(lambda x: (x * w3).sum(), raw=True)
             .rank(axis=1, pct=True))

    alpha = -(part1 - part2)
    return alpha

def alpha092(high, low, close, open, volume):
    """
    Alpha#92: min(Ts_Rank(decay_linear(((hl2+close) < (low+open)), 15), 19),
                  Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7))
    """
    adv30 = volume.rolling(30).mean()
    hl2   = (high + low) / 2

    w15 = np.arange(1, 16, dtype=float); w15 /= w15.sum()
    w7  = np.arange(1, 8,  dtype=float); w7  /= w7.sum()

    inner1 = ((hl2 + close) < (low + open)).astype(float)
    part1  = (inner1.rolling(15).apply(lambda x: (x * w15).sum(), raw=True)
              .rolling(19).rank(pct=True))
    part2  = (low.rank(axis=1, pct=True)
              .rolling(8).corr(adv30.rank(axis=1, pct=True))
              .rolling(7).apply(lambda x: (x * w7).sum(), raw=True)
              .rolling(7).rank(pct=True))

    alpha = pd.DataFrame(np.minimum(part1.values, part2.values),
                         index=close.index, columns=close.columns)
    return alpha

def alpha093(vwap, close, volume, industry):
    """
    Alpha#93: Ts_Rank(decay_linear(correlation(IndNeutralize(vwap,industry), adv81, 17), 20), 8)
              / rank(decay_linear(delta(close*0.524434+vwap*0.475566, 3), 16))
    """
    industry_s = industry['industry']
    adv81      = volume.rolling(81).mean()
    vwap_neu   = indneutralize(vwap, industry_s)
    price      = close * 0.524434 + vwap * (1 - 0.524434)

    w20 = np.arange(1, 21, dtype=float); w20 /= w20.sum()
    w16 = np.arange(1, 17, dtype=float); w16 /= w16.sum()

    part1 = (vwap_neu.rolling(17).corr(adv81)
             .rolling(20).apply(lambda x: (x * w20).sum(), raw=True)
             .rolling(8).rank(pct=True))
    part2 = (price.diff(3)
             .rolling(16).apply(lambda x: (x * w16).sum(), raw=True)
             .rank(axis=1, pct=True))

    alpha = part1 / part2
    return alpha

def alpha094(vwap, volume):
    """
    Alpha#94: (rank(vwap - ts_min(vwap,12))
               ^ Ts_Rank(correlation(Ts_Rank(vwap,20), Ts_Rank(adv60,4), 18), 3)) * -1
    """
    adv60      = volume.rolling(60).mean()
    rank_left  = (vwap - vwap.rolling(12).min()).rank(axis=1, pct=True)
    rank_right = (vwap.rolling(20).rank(pct=True)
                  .rolling(18).corr(adv60.rolling(4).rank(pct=True))
                  .rolling(3).rank(pct=True))

    alpha = -(rank_left ** rank_right)
    return alpha

def alpha095(open, high, low, volume):
    """
    Alpha#95: rank(open - ts_min(open,12))
              < Ts_Rank(rank(correlation(sum((high+low)/2,19), sum(adv40,19), 13))^5, 12)
    """
    adv40      = volume.rolling(40).mean()
    hl2        = (high + low) / 2
    rank_left  = (open - open.rolling(12).min()).rank(axis=1, pct=True)
    rank_right = ((hl2.rolling(19).sum()
                   .rolling(13).corr(adv40.rolling(19).sum())
                   .rank(axis=1, pct=True) ** 5)
                  .rolling(12).rank(pct=True))

    alpha = rank_right - rank_left
    return alpha

def alpha096(vwap, close, volume):
    """
    Alpha#96: max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4), 8),
                  Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close,7), Ts_Rank(adv60,4), 4), 13), 14), 13)) * -1
    """
    adv60 = volume.rolling(60).mean()

    w4 = np.arange(1, 5, dtype=float);
    w4 /= w4.sum()
    w14 = np.arange(1, 15, dtype=float);
    w14 /= w14.sum()

    # Part1
    part1 = (vwap.rank(axis=1, pct=True)
             .rolling(4).corr(volume.rank(axis=1, pct=True))
             .rolling(4).apply(lambda x: (x * w4).sum(), raw=True)
             .rolling(8).rank(pct=True))

    # Part2 - 修复 ts_argmax 的 NaN 处理
    close_rank = close.rolling(7).rank(pct=True)
    adv60_rank = adv60.rolling(4).rank(pct=True)
    corr2 = close_rank.rolling(4).corr(adv60_rank)

    # 关键修复：允许窗口内有 NaN，用 nanargmax
    def safe_argmax(x):
        arr = np.array(x, dtype=float)
        if np.all(np.isnan(arr)):
            return np.nan
        return float(np.nanargmax(arr))  # ← nanargmax 跳过 NaN

    ts_argmax = corr2.rolling(13, min_periods=5).apply(safe_argmax, raw=True)

    part2 = (ts_argmax
             .rolling(14, min_periods=7).apply(lambda x: (x * w14[:len(x)]).sum() / (w14[:len(x)].sum()), raw=True)
             .rolling(13, min_periods=5).rank(pct=True))

    alpha = -pd.DataFrame(np.maximum(part1.values, part2.values),
                          index=vwap.index, columns=vwap.columns)
    return alpha

def alpha097(low, vwap, volume, industry):
    """
    Alpha#97: (rank(decay_linear(delta(IndNeutralize(low*0.721001+vwap*0.278999, industry), 3), 20))
               - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,8), Ts_Rank(adv60,17), 5), 19), 16), 7)) * -1
    """
    industry_s = industry['industry']
    adv60      = volume.rolling(60).mean()
    price      = low * 0.721001 + vwap * (1 - 0.721001)
    price_neu  = indneutralize(price, industry_s)

    w20 = np.arange(1, 21, dtype=float); w20 /= w20.sum()
    w16 = np.arange(1, 17, dtype=float); w16 /= w16.sum()

    part1 = (price_neu.diff(3)
             .rolling(20).apply(lambda x: (x * w20).sum(), raw=True)
             .rank(axis=1, pct=True))
    part2 = (low.rolling(8).rank(pct=True)
             .rolling(5).corr(adv60.rolling(17).rank(pct=True))
             .rolling(19).rank(pct=True)
             .rolling(16).apply(lambda x: (x * w16).sum(), raw=True)
             .rolling(7).rank(pct=True))

    alpha = -(part1 - part2)
    return alpha

def alpha098(vwap, open, volume):
    """
    Alpha#98: rank(decay_linear(correlation(vwap, sum(adv5,26), 5), 7))
              - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15),21), 9), 7), 8))
    """
    adv5  = volume.rolling(5).mean()
    adv15 = volume.rolling(15).mean()

    w7 = np.arange(1, 8, dtype=float); w7 /= w7.sum()
    w8 = np.arange(1, 9, dtype=float); w8 /= w8.sum()

    part1 = (vwap.rolling(5).corr(adv5.rolling(26).sum())
             .rolling(7).apply(lambda x: (x * w7).sum(), raw=True)
             .rank(axis=1, pct=True))

    corr2     = open.rank(axis=1, pct=True).rolling(21).corr(adv15.rank(axis=1, pct=True))
    ts_argmin = corr2.rolling(9).apply(lambda x: int(np.argmin(x)), raw=True)
    part2     = (ts_argmin
                 .rolling(7).rank(pct=True)
                 .rolling(8).apply(lambda x: (x * w8).sum(), raw=True)
                 .rank(axis=1, pct=True))

    alpha = part1 - part2
    return alpha

def alpha099(high, low, close, volume):
    """
    Alpha#99: (rank(correlation(sum((high+low)/2,20), sum(adv60,20), 9))
               < rank(correlation(low, volume, 6))) * -1
    """
    adv60      = volume.rolling(60).mean()
    hl2        = (high + low) / 2
    rank_left  = (hl2.rolling(20).sum()
                  .rolling(9).corr(adv60.rolling(20).sum())
                  .rank(axis=1, pct=True))
    rank_right = low.rolling(6).corr(volume).rank(axis=1, pct=True)

    alpha = -(rank_right - rank_left)
    return alpha

def alpha100(close, high, low, volume, industry):
    """
    Alpha#100: 0 - (1.5*scale(indneutralize(indneutralize(rank(((close-low)-(high-close))/(high-low)*volume),
                               subindustry), subindustry))
                   - scale(indneutralize(correlation(close, rank(adv20),5) - rank(ts_argmin(close,30)),
                            subindustry))) * volume/adv20
    """
    industry_s = industry['industry']
    adv20      = volume.rolling(20).mean()

    inner      = ((close - low) - (high - close)) / (high - low) * volume
    rank_inner = inner.rank(axis=1, pct=True)
    neu1       = indneutralize(indneutralize(rank_inner, industry_s), industry_s)
    scale1     = neu1.div(neu1.abs().sum(axis=1), axis=0)

    corr       = close.rolling(5).corr(adv20.rank(axis=1, pct=True))
    ts_argmin  = close.rolling(30).apply(lambda x: int(np.argmin(x)), raw=True)
    neu2       = indneutralize(corr - ts_argmin.rank(axis=1, pct=True), industry_s)
    scale2     = neu2.div(neu2.abs().sum(axis=1), axis=0)

    alpha = -((1.5 * scale1 - scale2) * (volume / adv20))
    return alpha

def alpha101(close, open, high, low):
    """
    Alpha#101: (close - open) / ((high - low) + 0.001)
    """
    alpha = (close - open) / ((high - low) + 0.001)
    return alpha
