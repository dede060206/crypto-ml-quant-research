"""Offline Buy & Hold validation, with explicit proxy limitations."""
import json
from pathlib import Path
import pandas as pd
from backtest.cost_model import Costs
from backtest.engine import buy_and_hold
from backtest.metrics import metrics, yearly_returns

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'reports'/'buy_hold_validation'
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    curves = {}
    for asset in ('BTC','ETH','SOL'):
        folder = ROOT/'data'/'cache'/f'{asset}_USDT_USDT'
        candles = pd.read_parquet(folder/'ohlcv.parquet')
        funding = pd.read_parquet(folder/'funding.parquet')
        gaps = (candles.timestamp.diff().dropna() != 28_800_000).sum()
        if gaps:
            results[asset] = {'status':'excluded', 'reason':f'{gaps} candle gaps; no price interpolation; pending source verification'}
            continue
        results[asset] = {'status':'ok', 'scenarios':{}}
        scenarios = [('price_only_10pct',Costs(0,0),None,.1), ('execution_costs_10pct',Costs(),None,.1), ('execution_and_funding_proxy_10pct',Costs(),funding,.1), ('price_only_100pct_reference',Costs(0,0),None,1)]
        for label,cost,fees,fraction in scenarios:
            curve, accounting = buy_and_hold(candles,fees,fraction=fraction,costs=cost)
            curve.to_csv(output/f'{asset}_{label}.csv')
            curves[f'{asset} {label}'] = curve
            results[asset]['scenarios'][label] = {'metrics':metrics(curve), 'yearly_returns':yearly_returns(curve), 'accounting':accounting}
        results[asset]['slippage_sensitivity'] = {}
        for slip in (0,.0002,.0003,.0005,.001):
            curve, _ = buy_and_hold(candles,funding,fraction=.1,costs=Costs(.0005,slip))
            results[asset]['slippage_sensitivity'][str(slip)] = metrics(curve)['total_return']
        # Independent oracle: price-only fixed-units equity is exactly cash + asset value.
        p0 = candles.close.iloc[0]
        expected = 10000*(.9+.1*candles.close.to_numpy()/p0)
        import numpy as np
        np.testing.assert_allclose(curves[f'{asset} price_only_10pct'].equity.iloc[1:].to_numpy(),expected,rtol=1e-12)
    (output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    # Standalone interactive chart; all chart data embedded, no external service needed.
    chart = {}
    for name,curve in curves.items():
        daily = curve.equity.resample('1D').last()
        chart[name] = [[str(t.date()),round(float(v/10000),6)] for t,v in daily.items()]
    summary = '\n'.join(f"<tr><td>{asset}</td><td>{label}</td><td>{v['metrics']['total_return']:.2%}</td><td>{v['metrics']['max_drawdown']:.2%}</td><td>{v['metrics']['sharpe_daily_365']:.3f}</td></tr>" for asset,a in results.items() if a['status']=='ok' for label,v in a['scenarios'].items())
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Buy & Hold 回测验证</title><style>body{font:16px system-ui;margin:36px auto;max-width:1100px;padding:0 20px;color:#17324d}canvas{width:100%;height:420px;border:1px solid #ddd}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd}select{padding:8px}p{line-height:1.7}</style><h1>Buy & Hold 回测验证</h1><p>2021-01 至 2026-08 · 固定单位持仓 · 初始资金 10,000 USDT · 默认初始仓位 10%。本报告验证引擎，尚无机器学习策略结果。</p><p>手续费每侧0.05%，滑点每侧0.03%。资金费用使用实际结算费率和最近已收盘K线价格近似名义价值，缺少真实标记价格，因此含资金费用结果是近似值。100%价格参考使用永续价格代理BTC/ETH Buy & Hold，不是已验证的现货基准。SOL因行情缺口暂不运行。</p><label>查看曲线 <select id="choice"></select></label><canvas id="chart" width="1100" height="420"></canvas><p id="detail"></p><table><tr><th>资产</th><th>场景</th><th>累计收益</th><th>最大回撤</th><th>Sharpe</th></tr>SUMMARY</table><p>Sharpe/Sortino基于UTC日收益，365天年化，零无风险利率。首日包括入场成本；最后一日包含退出成本。回撤持续时间含尚未恢复的尾部。每个场景仅一笔交易，盈亏比和profit factor不作统计解释；按年收益首尾年份为部分年份。</p><script>const data=DATA;const select=document.getElementById('choice');Object.keys(data).forEach(x=>select.add(new Option(x,x)));function draw(){let a=data[select.value],c=document.getElementById('chart'),g=c.getContext('2d'),lo=Math.min(1,...a.map(x=>x[1])),hi=Math.max(1,...a.map(x=>x[1]));if(hi===lo)hi=lo+1;g.clearRect(0,0,c.width,c.height);const X=i=>65+i*(c.width-90)/(a.length-1),Y=v=>360-(v-lo)*320/(hi-lo);g.font='13px system-ui';g.fillStyle='#17324d';for(let j=0;j<=4;j++){let v=lo+(hi-lo)*j/4;g.fillText(v.toFixed(2)+'x',8,Y(v)+4);g.strokeStyle='#e3e9ef';g.beginPath();g.moveTo(65,Y(v));g.lineTo(1080,Y(v));g.stroke()}g.strokeStyle='#087f8c';g.lineWidth=2;g.beginPath();a.forEach((p,i)=>i?g.lineTo(X(i),Y(p[1])):g.moveTo(X(i),Y(p[1])));g.stroke();g.fillText(a[0][0],65,395);g.fillText(a.at(-1)[0],990,395);document.getElementById('detail').textContent=select.value+' · 期末净值 '+a.at(-1)[1].toFixed(4)+'x'}select.onchange=draw;draw();</script></html>'''
    (output/'report.html').write_text(html.replace('SUMMARY',summary).replace('DATA',json.dumps(chart)))
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
