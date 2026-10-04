"""Offline research dataset builder; no model fit and no global normalization."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from data.common import save
from .price_features import price_features, PRICE_COLUMNS
from .funding_features import align_funding, FUNDING_COLUMNS, funding_columns
from .target_construction import construct_targets, cost_threshold
from backtest.cost_model import Costs

ROOT=Path(__file__).resolve().parents[1]
FEATURE_COLUMNS=PRICE_COLUMNS+FUNDING_COLUMNS


def build_features(candles,funding,publication_lag_ms=60000,windows=None):
    return align_funding(price_features(candles),funding,publication_lag_ms,windows)


def build_dataset(candles,funding,publication_lag_ms=60000,horizon_bars=3,costs=Costs(),windows=None):
    features=build_features(candles,funding,publication_lag_ms,windows)
    labels=construct_targets(candles,horizon_bars,costs)
    result=features.merge(labels,on='timestamp',how='left',validate='one_to_one')
    result['features_valid']=np.isfinite(result[PRICE_COLUMNS+funding_columns(windows)]).all(axis=1)
    result['training_eligible']=result.features_valid & result.label_valid
    return result


def main():
    config=yaml.safe_load((ROOT/'configs/config.yaml').read_text())
    settings=config['research_dataset']
    output=ROOT/settings['output_dir']
    costs=Costs(settings['fee_per_side'],settings['slippage_per_side'])
    summary={'feature_columns':FEATURE_COLUMNS,'horizon_bars':settings['horizon_bars'],'cost_threshold':cost_threshold(costs),'publication_lag_ms':settings['funding_publication_lag_ms'],'assets':{}}
    for symbol in config['symbols']:
        slug=symbol.replace('/','_').replace(':','_')
        source=ROOT/config['cache_dir']/slug
        candles=pd.read_parquet(source/'ohlcv.parquet')
        funding=pd.read_parquet(source/'funding.parquet')
        dataset=build_dataset(candles,funding,settings['funding_publication_lag_ms'],settings['horizon_bars'],costs)
        # Explicit allowlist keeps label values, prices and identifiers out of future model X.
        save(dataset,output/slug/'all_samples.parquet')
        eligible=dataset[dataset.training_eligible].copy()
        use=symbol in settings['training_symbols']
        if use:
            save(eligible,output/slug/'training_samples.parquet')
        else:
            old=output/slug/'training_samples.parquet'
            if old.exists():old.unlink()
        asset={'candles':len(dataset),'invalid_features':int((~dataset.features_valid).sum()),'invalid_labels':int((~dataset.label_valid).sum()),'eligible_rows':len(eligible),'positive_label_fraction':float(eligible.target.mean()) if len(eligible) else None,'training_enabled':use,'first_signal_utc':pd.Timestamp(int(eligible.signal_time.min()),unit='ms',tz='UTC').isoformat() if len(eligible) else None,'last_signal_utc':pd.Timestamp(int(eligible.signal_time.max()),unit='ms',tz='UTC').isoformat() if len(eligible) else None}
        summary['assets'][symbol]=asset
    output.mkdir(parents=True,exist_ok=True)
    (output/'dataset_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
