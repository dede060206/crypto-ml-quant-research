"""Generate honest OOS strategy reports, QuantStats checks and executed notebook."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work'/'matplotlib'))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import quantstats as qs
from models.settings import settings
from .strategy import backtest_predictions
from .cost_model import Costs
from .metrics import metrics,yearly_returns,daily_returns

OUT=ROOT/'reports'/'ml_research'


def trade_metrics(trades):
    if trades.empty:return {'trade_count':0,'trade_win_rate':None,'trade_profit_factor':None,'trade_payoff_ratio':None}
    pnl=trades.net_pnl;wins=pnl[pnl>0];losses=pnl[pnl<0]
    return {'trade_count':len(trades),'trade_win_rate':float((pnl>0).mean()),'trade_profit_factor':float(wins.sum()/(-losses.sum())) if len(losses) else None,'trade_payoff_ratio':float(wins.mean()/(-losses.mean())) if len(wins) and len(losses) else None}


def spot_curve(spot,start,end,fraction=1):
    lookup=spot.set_index('timestamp')
    if start not in lookup.index or end not in lookup.index:raise ValueError('Spot reference boundaries missing')
    between=spot[(spot.timestamp>=start)&(spot.timestamp<end)]
    ts=np.r_[start-1,start,between.timestamp.to_numpy()+28_800_000-1,end]
    p0=float(lookup.loc[start,'open'])
    ps=np.r_[p0,p0,between.close.to_numpy(),float(lookup.loc[end,'open'])]
    return pd.DataFrame({'equity':10000*(1-fraction+fraction*ps/p0)},index=pd.to_datetime(ts,unit='ms',utc=True))


def main():
    cfg=settings()['strategy']
    options={'threshold':cfg['probability_threshold'],'fraction':cfg['entry_equity_fraction'],'capital':cfg['initial_capital']}
    predictions=pd.read_parquet(OUT/'oos_predictions.parquet')
    spot=pd.read_parquet(ROOT/'data'/'cache'/'BTC_SPOT'/'ohlcv.parquet')
    first=int(predictions.entry_time.min());last=int(predictions.label_end_time.max())
    benchmark=spot_curve(spot,first,last)
    benchmark.to_csv(OUT/'BTC_spot_buy_hold.csv')
    ref_returns=daily_returns(benchmark)
    summary={'benchmark_BTC_spot_100pct':metrics(benchmark),'benchmark_BTC_spot_10pct':metrics(spot_curve(spot,first,last,.1)),'strategies':{}}
    qs_checks=[];regimes=[];daily_year=[];curves={}
    btc=pd.read_parquet(ROOT/'data'/'cache'/'BTC_USDT_USDT'/'ohlcv.parquet')
    btc_daily=pd.Series(btc.close.to_numpy(),index=pd.to_datetime(btc.timestamp+28_800_000-1,unit='ms',utc=True)).resample('1D').last()
    past=btc_daily.pct_change(180).shift(1)
    regime=pd.Series(np.select([past>.2,past<-.2],['bull','bear'],default='sideways'),index=past.index)
    regime[past.isna()]='unavailable'
    for asset in settings()['assets']:
        root=ROOT/'data'/'cache'/f'{asset}_USDT_USDT'
        candles=pd.read_parquet(root/'ohlcv.parquet');funding=pd.read_parquet(root/'funding.parquet')
        for model in ['logreg_price','logreg_funding','xgb_price','xgb_funding','logreg_z_24h','logreg_z_3d','logreg_z_7d']+[prefix+'_windows_'+profile for profile in settings()['window_sensitivity'] for prefix in ['logreg','xgb']]:
            p=predictions[(predictions.asset==asset)&(predictions.model==model)]
            for scenario,cost,f in [('gross',Costs(0,0),None),('execution',Costs(cfg['fee_per_side'],cfg['slippage_per_side']),None),('net',Costs(cfg['fee_per_side'],cfg['slippage_per_side']),funding)]:
                curve,trades=backtest_predictions(p,candles,f,cost,**options)
                key=asset+'_'+model+'_'+scenario
                m=metrics(curve);m.update(trade_metrics(trades))
                summary['strategies'][key]={'metrics':m,'yearly_returns':yearly_returns(curve),'fees_paid':float(trades.fees.sum()) if len(trades) else 0,'slippage_paid':float(trades.slippage_cost.sum()) if len(trades) else 0,'funding_paid_net':float(trades.funding_paid_net.sum()) if len(trades) else 0}
                curve.to_csv(OUT/(key+'_equity.csv'));trades.to_csv(OUT/(key+'_trades.csv'),index=False)
                curves[key]=curve
                if scenario=='net':
                    ret=daily_returns(curve)
                    for label in ['bull','bear','sideways']:
                        subset=ret[regime.reindex(ret.index)==label]
                        regimes.append({'strategy':key,'regime':label,'days':len(subset),'mean_daily_return':float(subset.mean()),'positive_day_fraction':float((subset>0).mean()),'sharpe_365':float(subset.mean()/subset.std(ddof=1)*np.sqrt(365)) if subset.std(ddof=1)>0 else None})
                    if model in ['logreg_funding','xgb_funding']:
                        checks={'strategy':key,'our_sharpe':m['sharpe_daily_365'],'qs_sharpe':float(qs.stats.sharpe(ret,periods=365)),'our_sortino':m['sortino_daily_365'],'qs_sortino':float(qs.stats.sortino(ret,periods=365))}
                        daily_eq=10000*(1+ret).cumprod();dd=(daily_eq/np.maximum.accumulate(np.r_[10000,daily_eq.to_numpy()])[1:]-1).min()
                        checks['our_daily_max_drawdown']=float(dd);checks['qs_daily_max_drawdown']=float(qs.stats.max_drawdown(ret))
                        for a,b in [('our_sharpe','qs_sharpe'),('our_sortino','qs_sortino'),('our_daily_max_drawdown','qs_daily_max_drawdown')]:np.testing.assert_allclose(checks[a],checks[b],rtol=1e-9,atol=1e-10)
                        qs_checks.append(checks)
                        qs_ret=ret.copy();qs_ref=ref_returns.copy()
                        qs_ret.index=qs_ret.index.tz_localize(None).as_unit('ns')
                        qs_ref.index=qs_ref.index.tz_localize(None).as_unit('ns')
                        qs.reports.html(qs_ret,benchmark=qs_ref,periods_per_year=365,rf=0,title=asset+' '+model+' OOS (funding price proxy)',output=str(OUT/(key+'_quantstats.html')),match_dates=False)
                print(key,'trades',len(trades),'return',m['total_return'],flush=True)
            sensitivity={}
            for slip in [0,.0002,.0003,.0005,.001]:
                curve,_=backtest_predictions(p,candles,funding,Costs(cfg["fee_per_side"],slip),**options)
                sensitivity[str(slip)]=metrics(curve)['total_return']
            summary['strategies'][asset+'_'+model+'_net']['slippage_sensitivity']=sensitivity
    pd.DataFrame(regimes).to_csv(OUT/'regime_metrics.csv',index=False)
    (OUT/'quantstats_crosscheck.json').write_text(json.dumps(qs_checks,indent=2,allow_nan=False))
    (OUT/'backtest_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    fig,axes=plt.subplots(len(settings()['assets']),1,figsize=(12,11),sharex=True)
    for ax,asset in zip(axes,settings()['assets']):
        for model in ['logreg_price','logreg_funding','xgb_price','xgb_funding']:
            daily=curves[asset+'_'+model+'_net'].equity.resample('1D').last()
            ax.plot(daily.index,daily/10000,label=model)
        ax.plot(benchmark.index,1+.1*(benchmark.equity/10000-1),label='BTC spot B&H (10% initial)',linestyle='--',color='black',alpha=.6)
        ax.set_title(asset+' OOS: 10% entry equity, long/cash, net costs + funding proxy');ax.set_ylabel('Equity multiple');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'oos_equity.png',dpi=160);plt.close(fig)
    folds=pd.read_csv(OUT/'fold_metrics.csv')
    fig,axes=plt.subplots(len(settings()['assets']),1,figsize=(12,10),sharex=True)
    for ax,asset in zip(axes,settings()['assets']):
        for model in ['logreg_price','logreg_funding','xgb_price','xgb_funding']:
            part=folds[(folds.asset==asset)&(folds.model==model)]
            ax.plot(pd.to_datetime(part.fold),part.roc_auc,label=model)
        ax.axhline(.5,color='black',linestyle='--');ax.set_title(asset+' monthly OOS ROC AUC');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'monthly_auc.png',dpi=160);plt.close(fig)
    write_report(summary)


def write_report(backtest):
    summary=json.loads((OUT/'classification_summary.json').read_text());delta=json.loads((OUT/'funding_incremental_value.json').read_text())
    lines=['# Crypto ML Quant：滚动样本外研究报告','', '## 研究结论','', '本次固定参数研究未发现资金费率特征带来稳定的方向预测增益。不能根据某个正收益场景宣称信号有效。报告保留表现不佳的结果，不使用样本外收益反向调参。','', '## 验证设计','', 'BTC、ETH、SOL分别训练，2022-01至2026-08，共56个月度测试折／资产。训练12个日历月，每次前推1个月；按label_end_time剔除与测试边界重叠的标签，并额外隔离8小时。每折重新拟合训练集标准化器；XGBoost不用外层测试集early stopping。逻辑回归C=1；XGBoost150树、深度3、学习率0.03、min_child_weight=10、lambda=5、subsample/colsample=0.8，随机种子42。主模型参数在最初查看样本外结果前固定；本次新增窗口长度与SOL分析属于收尾扩展探索，未用来选择最佳模型。','', '13项特征及24小时标签见../DATASET_METHOD.md。资金费率60秒可用延迟是研究假设，不是观测到的真实公布时间；下一根开盘执行是理想化零计算延迟。单次2025-09固定切分仅检验流程，不是与滚动结果独立的额外证据。','', '## 分类表现','', '|资产|模型|样本数|ROC AUC|Log loss|Brier|','|---|---|---:|---:|---:|---:|']
    for asset in settings()['assets']:
        for model in ['dummy','logreg_price','logreg_funding','xgb_price','xgb_funding']:
            m=summary[asset+'_'+model];lines.append(f"|{asset}|{model}|{m['n']}|{m['roc_auc']:.4f}|{m['log_loss']:.4f}|{m['brier']:.4f}|")
    lines+=['','模型名称funding表示价格加资金费率，并非仅资金费率。Dummy使用各折训练集阳性比例。AUC约0.5代表排序能力接近随机；Log loss越低越好。每折训练与测试损失、AUC、AP、类别比例、样本数与purge数量完整保留在fold_metrics.csv。','', '## 资金费率增量证据','', '|资产|比较|平均月度Log loss差值|月块bootstrap 95%区间|','|---|---|---:|---|']
    for x in delta:
        lo,hi=x['month_block_bootstrap_95pct'];lines.append(f"|{x['asset']}|{x['comparison']}|{x['mean_monthly_logloss_delta']:.5f}|[{lo:.5f}, {hi:.5f}]|")
    lines+=['','正差值表示加入资金费率后更差。月块bootstrap考虑24小时标签重叠导致的样本依赖，但相邻月份仍可能相关，区间并非严格显著性结论。','', '## 交易规则与回测','', '概率>=0.55时做多，其他时候空仓；每次用当时权益10%的预算，手续费包含在预算内。固定单位持有24小时，只允许一笔活动交易，持仓期间其他信号忽略。退出与下一次入场可在同一开盘边界发生。BTC/ETH/SOL各独立10,000 USDT账户，不是合成投资组合。手续费每侧0.05%，滑点每侧0.03%；资金费率按实际结算时间支付，名义价值用最近已收盘价格代理真实标记价格。无成本版本也不含资金费用，execution版本仅手续费滑点，net版本再计资金费用。','', '|资产|模型|交易数|无成本收益|净收益（近似）|净最大回撤|净Sharpe|胜率|','|---|---|---:|---:|---:|---:|---:|---:|']
    for asset in settings()['assets']:
        for model in ['logreg_price','logreg_funding','xgb_price','xgb_funding']:
            g=backtest['strategies'][asset+'_'+model+'_gross']['metrics'];m=backtest['strategies'][asset+'_'+model+'_net']['metrics'];lines.append(f"|{asset}|{model}|{m['trade_count']}|{g['total_return']:.2%}|{m['total_return']:.2%}|{m['max_drawdown']:.2%}|{m['sharpe_daily_365']:.3f}|{m['trade_win_rate']:.2%}|")
    bm=backtest['benchmark_BTC_spot_100pct'];b10=backtest['benchmark_BTC_spot_10pct'];lines+=['',f"同一持有区间真实BTC现货Buy & Hold（100%投入、无成本）收益{bm['total_return']:.2%}；10%初始投入、固定单位参考收益{b10['total_return']:.2%}。现货基准无持仓资金费；100%基准与10%策略风险预算不同，不能直接用绝对收益判断模型优劣。现货源ZIP已按官方校验，2025起微秒时间戳已显式转换为毫秒。",'', '## 稳健性与限制','', '- BTC/ETH/SOL跨资产、价格对照、逻辑回归/XGBoost对照，以及单个资金z-score特征子集均完整保存。另实际重算短窗口组合24h/2d/5d及长窗口组合48h/5d/14d，对逻辑回归和XGBoost均重训全部56折，使用相同样本池。窗口实验属于探索扩展，所有结果保留，不据此挑选策略，也不冒充独立确认数据。','- 逐年收益、滑点0–0.1%敏感性、Sortino、回撤持续时间、盈亏比和profit factor见backtest_summary.json。盈亏比为平均盈利／平均亏损绝对值；profit factor为总盈利／总亏损绝对值。','- 市场阶段使用BTC此前180天价格收益，>20%为bull、<-20%为bear，其余sideways；阶段在当日开始前已可用。regime_metrics.csv报告条件日均收益及Sharpe，不把非连续阶段拼接计算回撤。阶段定义是研究约定。','- 每折结果见fold_metrics.csv，不能只依据聚合指标判断稳定性。训练损失低于测试损失可能反映过拟合与市场变化；当前信号弱、历史状态变化和样本有限均可能解释差结果。','- SOL月度文件缺失15根8h K线，已由官方每日8h文件的真实记录补齐；相邻日期每日/月度重叠OHLCV一致，原始缺失时间与校验来源保存在gap_audit.json。结算间隔变化仍按真实时间处理，异常边界重启资金统计。选择现存主流资产仍存在生存者偏差。','- 资金成本缺标记价格、未实现真实保证金清算／盘口冲击／报单延迟；10%仓位用于控制暴露，结果仍为研究性近似，不是可实盘执行证明。','- 因研究者已查看全部样本外结果，后续任何改动必须记录为探索；严格确认新假设需要新的封存数据。','- QuantStats的日Sharpe/Sortino/日最大回撤已与手写公式逐项核对；日最大回撤与8h事件权益回撤可能不同，不应混为同一指标。','', '## 结果文件','', 'oos_predictions.parquet保留每折概率与时间；每个策略的_equity.csv与_trades.csv可复核。6份QuantStats HTML及两张图、结构化结果一并提供。','', '![OOS equity](oos_equity.png)','', '![Monthly AUC](monthly_auc.png)','', '## 重现','', '```sh','./run_research.sh','```','']
    text='\n'.join(lines);(OUT/'RESEARCH_REPORT.md').write_text(text)
    import nbformat
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text.replace('![OOS equity](oos_equity.png)','').replace('![Monthly AUC](monthly_auc.png)','')),nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image\nroot = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nout = root / 'reports' / 'ml_research'\ndisplay(pd.read_csv(out / 'fold_metrics.csv'))\ndisplay(Image(filename=str(out / 'oos_equity.png')))\ndisplay(Image(filename=str(out / 'monthly_auc.png')))\nchecks = json.loads((out / 'quantstats_crosscheck.json').read_text())\ndisplay(pd.DataFrame(checks))")])
    notebook.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
    (ROOT/'notebooks').mkdir(exist_ok=True)
    nbformat.write(notebook,ROOT/'notebooks'/'research_report.ipynb')
    # Static HTML and executed notebook with saved tables/images; no need to rerun training.
    from nbclient import NotebookClient
    client=NotebookClient(notebook,timeout=60,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.execute();nbformat.write(notebook,ROOT/'notebooks'/'research_report.ipynb')
    from nbconvert import HTMLExporter
    body,_=HTMLExporter().from_notebook_node(notebook);(OUT/'research_report.html').write_text(body)


if __name__=='__main__':main()
