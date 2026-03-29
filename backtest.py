# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Project Name : 低价股回测脚本
# Author       : Nidol
# Date         : 2026-02-05
# Description  : 
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import glob
import warnings
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# 忽略 Pandas 的 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# ================= 配置区域 =================
DATA_DIR = 'stock'        # 股票数据目录
BENCHMARK_FILE = 'sh000300.csv' # 沪深300指数文件
MIN_PRICE = 3.0           # 最小价格限制
HOLD_COUNT = 5          # 持仓数量
COST_RATE = 0.0006         # 单边交易成本 (千分之一)
REBALANCE_WEEKDAY = 4     # 换仓日：0=周一, 4=周五
# ===========================================

def load_data(data_dir, verbose=True):
    """
    加载所有股票数据，并合并为一个大的 DataFrame (Pivot Table)
    """
    if verbose:
        print("正在加载股票数据...")
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not files:
        if verbose:
            print("错误：未找到股票数据文件。")
        return None, None, None

    def read_file(file_path):
        try:
            df = pd.read_csv(file_path, usecols=['ts_code', 'date', 'close', 'pre_close', 'amount'])
            return df
        except Exception:
            return None

    all_dfs = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(read_file, files)
        for i, df in enumerate(results):
            if df is not None:
                all_dfs.append(df)
            if verbose and (i + 1) % 500 == 0:
                print(f"  已处理 {i + 1} 个文件...")

    if not all_dfs:
        if verbose:
            print("错误：未能加载任何有效数据。")
        return None, None, None

    if verbose:
        print("正在合并数据...")
    full_df = pd.concat(all_dfs)
    full_df['date'] = pd.to_datetime(full_df['date'])
    full_df.drop_duplicates(subset=['date', 'ts_code'], inplace=True)

    if verbose:
        print("正在构建透视表...")
    close_df = full_df.pivot(index='date', columns='ts_code', values='close')
    pre_close_df = full_df.pivot(index='date', columns='ts_code', values='pre_close')
    amount_df = full_df.pivot(index='date', columns='ts_code', values='amount')
    
    close_df = close_df.sort_index()
    pre_close_df = pre_close_df.sort_index()
    amount_df = amount_df.sort_index()
    
    if verbose:
        print(f"数据加载完成。时间范围: {close_df.index.min().date()} 至 {close_df.index.max().date()}")
        print(f"股票数量: {close_df.shape[1]}")
    
    return close_df, pre_close_df, amount_df

def load_benchmark(file_path):
    """加载基准指数 (沪深300)"""
    if not os.path.exists(file_path):
        print(f"警告：基准文件 {file_path} 不存在。")
        return None
    
    try:
        df = pd.read_csv(file_path)
        # 统一列名
        cols = {c.lower(): c for c in df.columns}
        date_col = None
        for candidate in ['date', 'trade_date', 'day']:
            if candidate in cols:
                date_col = cols[candidate]
                break
        
        if not date_col:
            date_col = df.columns[0] # 盲猜第一列
            
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        
        close_col = None
        for candidate in ['close', 'open']: # 优先 Close
             if candidate in cols:
                close_col = cols[candidate]
                break
        
        if not close_col:
             close_col = df.columns[1]

        return df[close_col]
    except Exception as e:
        print(f"读取基准文件出错: {e}")
        return None

def calculate_metrics(returns):
    """计算回测指标"""
    if returns.empty:
        return {}, pd.Series()

    # 1. 年化收益率 (假设252个交易日)
    annual_return = returns.mean() * 252
    
    # 2. 累计收益率
    cum_returns = (1 + returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    
    # 3. 最大回撤
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 4. 夏普比率 (无风险利率假设为0)
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    # 5. 卡玛比率
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    metrics = {
        '年化收益率': f"{annual_return:.2%}",
        '累计收益率': f"{total_return:.2%}",
        '最大回撤': f"{max_drawdown:.2%}",
        '夏普比率': f"{sharpe_ratio:.2f}",
        '卡玛比率': f"{calmar_ratio:.2f}"
    }
    
    return metrics, cum_returns

def filter_universe(close_df, pre_close_df, amount_df, ts_codes):
    """若 ts_codes 非空，仅保留这些标的列；列不存在则跳过。"""
    if not ts_codes:
        return close_df, pre_close_df, amount_df
    cols = [c for c in ts_codes if c in close_df.columns]
    if not cols:
        return None, None, None
    return close_df[cols], pre_close_df[cols], amount_df[cols]


def run_backtest(
    close_df,
    pre_close_df,
    amount_df,
    benchmark_series,
    *,
    min_price=None,
    hold_count=None,
    cost_rate=None,
    rebalance_weekday=None,
    amount_rank_pct=None,
    verbose=True,
    save_selection_path=None,
):
    """低价股策略回测。未传参数时使用模块顶部全局默认配置。"""
    min_price = MIN_PRICE if min_price is None else min_price
    hold_count = HOLD_COUNT if hold_count is None else hold_count
    cost_rate = COST_RATE if cost_rate is None else cost_rate
    rebalance_weekday = REBALANCE_WEEKDAY if rebalance_weekday is None else rebalance_weekday
    amount_rank_pct = 0.03 if amount_rank_pct is None else amount_rank_pct

    if verbose:
        print("\n开始回测...")
    
    # 对齐日期
    if benchmark_series is not None:
        common_dates = close_df.index.intersection(benchmark_series.index)
        close_df = close_df.loc[common_dates]
        pre_close_df = pre_close_df.loc[common_dates]
        amount_df = amount_df.loc[common_dates]
        benchmark_series = benchmark_series.loc[common_dates]
    else:
        if verbose:
            print("未提供基准数据，将只计算策略收益。")
    
    all_days = close_df.index
    if len(all_days) == 0:
        if verbose:
            print("没有重叠的交易日期。")
        return None

    # 确定换仓日: 每周的 rebalance_weekday (例如周五)
    is_rebalance_day = all_days.weekday == rebalance_weekday
    
    if verbose:
        print("计算股票上市日期 (用于剔除新股)...")
    # 优化：直接用 argmax (对于全NaN列可能会错，但这里股票肯定有数据)
    # 或者用 close_df.notna().idxmax()
    first_valid_date = close_df.notna().idxmax()
    data_start_date = all_days[0]
    
    # 初始化回测状态
    portfolio_value = 1.0
    portfolio_history = [] # [value]
    dates_history = []
    
    # 持仓字典: {ts_code: market_value}
    current_holdings = {} 
    
    # 记录选股结果
    selection_records = []
    
    if verbose:
        print(f"回测区间: {all_days[0].date()} 至 {all_days[-1].date()}")
    
    for i, today in enumerate(all_days):
        # 1. 计算当日持仓收益 (基于 PreClose -> Close)
        # 注意：这里的收益是 T日 相对 T-1日收盘(即 T日 PreClose) 的收益
        # 如果是连续持仓，T-1日的 Close 应该等于 T日的 PreClose (除权除息除外)
        # 既然数据提供了 PreClose，我们就用它来计算最准确的日内涨跌幅
        
        daily_pnl = 0.0
        
        if current_holdings:
            # 获取持仓股票的今日数据
            codes = list(current_holdings.keys())
            today_closes = close_df.loc[today, codes]
            today_pre_closes = pre_close_df.loc[today, codes]
            
            # 过滤掉今日无数据的股票 (停牌) - 保持价值不变
            valid_mask = today_closes.notna() & today_pre_closes.notna()
            valid_codes = today_closes.index[valid_mask]
            
            # 更新有效股票的价值
            # New Value = Old Value * (Close / PreClose)
            # 注意：这里的 Old Value 是基于上一日 Close 的吗？
            # 这里的逻辑假设：current_holdings 存储的是“上一日收盘时的市值”
            # 那么今日收盘市值 = 上一日收盘市值 * (TodayClose / TodayPreClose) ???
            # 只有当 TodayPreClose == YesterdayClose 时成立。
            # 如果发生分红除权，TodayPreClose < YesterdayClose，但股价下跌幅度也相应调整，
            # 真实的涨跌幅确实是 Close / PreClose - 1。
            # 所以：Value_T = Value_{T-1} * (Close_T / PreClose_T) 是正确的，因为它反映了复权后的真实增长。
            
            ratios = today_closes[valid_mask] / today_pre_closes[valid_mask]
            
            for code in valid_codes:
                current_holdings[code] *= ratios[code]
        
        # 计算今日总资产
        portfolio_value = sum(current_holdings.values()) if current_holdings else portfolio_value
        # 如果空仓，资金也应该有无风险收益？这里假设为0 (现金)
        
        # 记录净值
        portfolio_history.append(portfolio_value)
        dates_history.append(today)
        
        # 2. 换仓逻辑 (尾盘)
        if is_rebalance_day[i]:
            # --- 选股 ---
            # a. 基础池：今日有交易的股票
            valid_series = close_df.loc[today].dropna()
            if valid_series.empty:
                continue
                
            valid_stocks_idx = valid_series.index
            
            # b. 剔除上市不满一年 (365天)
            # 如果 first_valid_date == data_start_date，视为老股
            one_year_ago = today - pd.Timedelta(days=365)
            # 筛选条件： (上市日期 <= 一年前) OR (上市日期 == 数据起始日)
            is_old = (first_valid_date[valid_stocks_idx] <= one_year_ago) | \
                     (first_valid_date[valid_stocks_idx] == data_start_date)
            candidate_pool = valid_stocks_idx[is_old]
            
            # c. 价格过滤 (> min_price)
            current_prices = valid_series.loc[candidate_pool]
            candidate_pool = current_prices[current_prices > min_price].index

            # d. 成交额排名过滤 
            # 获取候选池中股票的成交额
            current_amounts = amount_df.loc[today, candidate_pool]
            # 计算排名百分比 (从小到大)
            # method='min' means same values get same rank. pct=True returns 0.0 to 1.0
            amount_ranks = current_amounts.rank(pct=True, ascending=True)
            rank_mask = (amount_ranks < amount_rank_pct)
            candidate_pool = amount_ranks[rank_mask].index
            
            if len(candidate_pool) == 0:
                # 无股可选，清仓?
                if current_holdings:
                     portfolio_value *= (1 - cost_rate)
                     current_holdings = {}
            else:
                # e. 策略：价格最低的 hold_count 只
                final_prices = valid_series.loc[candidate_pool]
                target_stocks = final_prices.nsmallest(hold_count).index.tolist()
                
                # 记录选股详细信息
                for stock in target_stocks:
                    selection_records.append({
                        'date': today,
                        'ts_code': stock,
                        'close': final_prices.loc[stock],
                        'amount': current_amounts.loc[stock] if stock in current_amounts.index else None,
                        'rank_pct': amount_ranks.loc[stock] if stock in amount_ranks.index else None
                    })
                
                # --- 调仓执行 ---
                # 计算交易成本
                old_codes = set(current_holdings.keys())
                new_codes = set(target_stocks)
                
                # 保留的股票 (不产生交易成本)
                keep_codes = old_codes & new_codes
                
                # 卖出部分
                sell_codes = old_codes - keep_codes
                sell_value = sum(current_holdings[c] for c in sell_codes)
                
                # 买入部分
                buy_codes = new_codes - keep_codes
                
                # 简化计算：
                # 假设我们将总资产重新平均分配给 new_codes
                # 那么卖出金额 = sum(current_holdings[c] for c in sell_codes)
                # 加上 需要减持的保留股票金额? 或者增持?
                # 为了简化：假设每次换仓都先全卖再全买 (成本会高估)
                # 优化模型：Cost = Sell_Cost + Buy_Cost
                # 目标：每个新股票分配 Value / N
                # 对于保留的股票：如果是增持，付买入费；减持付卖出费。
                
                target_val_per_stock = portfolio_value / len(target_stocks)
                
                total_cost = 0.0
                
                # 1. 卖出不再持有的
                total_cost += sell_value * cost_rate
                
                # 2. 调整保留的
                for code in keep_codes:
                    diff = target_val_per_stock - current_holdings[code]
                    total_cost += abs(diff) * cost_rate
                
                # 3. 买入新加的
                # Buy Value = len(buy_codes) * target_val_per_stock
                # 但这里不需要单独算，因为总资金是固定的，diff 已经涵盖了资金流动
                # 实际上：Sum(Diff) + Sell_Val = Buy_Val_Needed
                # 简单估算：Turnover Value = (Abs(Diff) Sum + Sell_Val + Buy_Val) / 2
                
                # 重新计算一次最简单的：
                # 新持仓总价值应该 = 旧总价值 - 成本
                # 设成本为 C
                # New_Total = Old_Total - C
                # 每个股票分 (Old_Total - C) / N
                # 这是一个方程，解起来麻烦。
                # 近似：C = Old_Total * Turnover_Rate * COST_RATE
                # Turnover Rate 约等于 (不再持有的比例 + 新进的比例) / 2 ?
                # 还是用简单的：非重合部分全部换手。
                turnover_ratio = (len(target_stocks) - len(keep_codes)) / len(target_stocks)
                # 假设双边成本
                cost = portfolio_value * turnover_ratio * cost_rate * 2
                
                portfolio_value -= cost
                
                # 更新持仓
                if portfolio_value > 0:
                    per_stock_val = portfolio_value / len(target_stocks)
                    current_holdings = {code: per_stock_val for code in target_stocks}
                else:
                    current_holdings = {}

    strategy_curve = pd.Series(portfolio_history, index=dates_history)
    strategy_returns = strategy_curve.pct_change().fillna(0)

    selection_df = pd.DataFrame(selection_records) if selection_records else pd.DataFrame()

    if save_selection_path is not None and not selection_df.empty:
        try:
            selection_df.to_csv(save_selection_path, index=False, encoding='utf-8-sig')
            if verbose:
                print(f"\n选股结果已保存至: {save_selection_path}")
        except Exception as e:
            if verbose:
                print(f"\n保存选股结果失败: {e}")
    elif verbose and not selection_records:
        print("\n未产生选股记录。")

    strat_metrics, strat_cum = calculate_metrics(strategy_returns)

    bench_metrics = None
    bench_cum = None
    if benchmark_series is not None:
        benchmark_series = benchmark_series.reindex(strategy_curve.index, method='ffill')
        bench_returns = benchmark_series.pct_change().fillna(0)
        bench_metrics, bench_cum = calculate_metrics(bench_returns)

    if verbose:
        print("\n---------- 回测结果 ----------")
        print("【低价股策略】")
        for k, v in strat_metrics.items():
            print(f"  {k}: {v}")
        if bench_metrics is not None:
            print("\n【沪深300基准】")
            for k, v in bench_metrics.items():
                print(f"  {k}: {v}")
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(strat_cum, label='Low Price Strategy')
            if bench_cum is not None:
                plt.plot(bench_cum, label='HS300 Benchmark', alpha=0.7)
            plt.title('Backtest Result: Low Price Strategy vs HS300')
            plt.legend()
            plt.grid(True)
            plt.savefig('backtest_result.png')
            plt.close()
            print("\n结果图表已保存为 backtest_result.png")
        except Exception as e:
            print(f"绘图失败: {e}")

    return {
        'strategy_metrics': strat_metrics,
        'strategy_cumulative': strat_cum,
        'strategy_curve': strategy_curve,
        'selection_df': selection_df,
        'bench_metrics': bench_metrics,
        'bench_cumulative': bench_cum,
    }


if __name__ == "__main__":
    print(f"数据目录: {DATA_DIR}")
    print(f"策略配置: 每周{REBALANCE_WEEKDAY+1}换仓, 持仓{HOLD_COUNT}只, 价格>{MIN_PRICE}")
    print("策略逻辑: 剔除新股/停牌 -> 价格>2元 -> 成交额排名最低3% -> 价格最低100只")
    print("注意: 暂未剔除 ST 股票 (数据源未包含状态标识)。")
    
    # 加载数据
    close_df, pre_close_df, amount_df = load_data(DATA_DIR)
    
    if close_df is not None:
        # 加载基准
        bench_series = load_benchmark(BENCHMARK_FILE)
        
        run_backtest(
            close_df,
            pre_close_df,
            amount_df,
            bench_series,
            save_selection_path='weekly_selections.csv',
        )
