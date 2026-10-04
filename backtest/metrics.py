"""Calendar-aware daily metrics, 365-day crypto year, zero risk-free return."""
import numpy as np
import pandas as pd


def daily_returns(curve):
    daily = curve.equity.resample('1D').last().ffill()
    returns = daily.pct_change()
    returns.iloc[0] = daily.iloc[0] / curve.equity.iloc[0] - 1
    return returns


def metrics(curve):
    eq = curve.equity
    returns = daily_returns(curve)
    std = returns.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(returns, 0)**2))
    dd = eq/eq.cummax()-1
    duration = pd.Timedelta(0)
    peak_time = eq.index[0]
    peak = eq.iloc[0]
    underwater = False
    for t, value in eq.items():
        if value >= peak:
            if underwater:
                duration = max(duration, t-peak_time)
            underwater = False
            peak, peak_time = value, t
        else:
            underwater = True
            duration = max(duration, t-peak_time)
    years = (eq.index[-1]-eq.index[0]).total_seconds()/(365*86400)
    return {'total_return':float(eq.iloc[-1]/eq.iloc[0]-1), 'cagr':float((eq.iloc[-1]/eq.iloc[0])**(1/years)-1) if years else None, 'sharpe_daily_365':float(returns.mean()/std*np.sqrt(365)) if std > 0 else None, 'sortino_daily_365':float(returns.mean()/downside*np.sqrt(365)) if downside > 0 else None, 'max_drawdown':float(dd.min()), 'max_drawdown_duration_days':duration.total_seconds()/86400, 'positive_day_fraction':float((returns>0).mean()), 'trade_win_rate':float(eq.iloc[-1]>eq.iloc[0]), 'trade_profit_factor':None, 'trade_payoff_ratio':None, 'trade_count':1}


def yearly_returns(curve):
    daily = curve.equity.resample('1D').last().ffill()
    previous = float(curve.equity.iloc[0])
    result = {}
    for year, segment in daily.groupby(daily.index.year):
        ending = float(segment.iloc[-1])
        result[str(year)] = ending/previous-1
        previous = ending
    return result
