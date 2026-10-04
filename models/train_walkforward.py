import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss,balanced_accuracy_score
from sklearn.dummy import DummyClassifier
from features.build_dataset import FEATURE_COLUMNS,build_dataset
from features.funding_features import funding_columns
from features.price_features import PRICE_COLUMNS
from .settings import settings
from .baseline_logreg import baseline_logreg
from .xgboost_model import xgboost_model
from evaluation.walk_forward_split import walk_forward_split

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'ml_research'


def classification_metrics(y,p):
    two=len(np.unique(y))==2
    return {'n':len(y),'positive_fraction':float(np.mean(y)),'roc_auc':float(roc_auc_score(y,p)) if two else None,'average_precision':float(average_precision_score(y,p)) if two else None,'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p)),'balanced_accuracy_at_05':float(balanced_accuracy_score(y,p>=.5)) if two else None}


def variants():
    result={'dummy':(lambda:DummyClassifier(strategy='prior'),PRICE_COLUMNS),'logreg_price':(baseline_logreg,PRICE_COLUMNS),'logreg_funding':(baseline_logreg,FEATURE_COLUMNS),'xgb_price':(xgboost_model,PRICE_COLUMNS),'xgb_funding':(xgboost_model,FEATURE_COLUMNS)}
    for window in ['24h','3d','7d']:
        cols=[c for c in FEATURE_COLUMNS if 'zscore' not in c or c.endswith('_'+window)]
        result['logreg_z_'+window]=(baseline_logreg,cols)
    return result


def main():
    cfg=settings()
    OUT.mkdir(parents=True,exist_ok=True)
    from hashlib import sha256
    import importlib.metadata as metadata
    paths=[ROOT/'data'/'processed'/f'{asset}_USDT_USDT'/'training_samples.parquet' for asset in cfg['assets']]
    spot=ROOT/'data'/'cache'/'BTC_SPOT'/'ohlcv.parquet'
    if spot.exists():paths.append(spot)
    record={'config':cfg,'input_sha256':{str(path.relative_to(ROOT)):sha256(path.read_bytes()).hexdigest() for path in paths},'versions':{name:metadata.version(name) for name in ['scikit-learn','xgboost','quantstats','pandas','numpy']},'model_selection':'none; all fixed variants retained','SOL_status':'monthly gaps repaired from verified daily 8h archives; provenance retained; funding interval anomaly resets funding statistics'}
    (OUT/'run_metadata.json').write_text(json.dumps(record,indent=2))
    all_folds=[];summary={};predictions=[];importance=[];holdout=[]
    for asset in cfg['assets']:
        d=pd.read_parquet(ROOT/'data'/'processed'/f'{asset}_USDT_USDT'/'training_samples.parquet').reset_index(drop=True)
        datasets={name:d for name in variants()}
        configs=variants()
        for profile,windows in cfg['window_sensitivity'].items():
            folder=ROOT/'data'/'cache'/f'{asset}_USDT_USDT'
            alternate=build_dataset(pd.read_parquet(folder/'ohlcv.parquet'),pd.read_parquet(folder/'funding.parquet'),windows=windows)
            alternate=alternate[alternate.training_eligible].reset_index(drop=True)
            # Require identical observation pools: window comparisons must not pick easier samples.
            np.testing.assert_array_equal(alternate.signal_time.to_numpy(),d.signal_time.to_numpy())
            cols=PRICE_COLUMNS+funding_columns(windows)
            for prefix,factory in [('logreg',baseline_logreg),('xgb',xgboost_model)]:
                name=prefix+'_windows_'+profile
                configs[name]=(factory,cols);datasets[name]=alternate
        for name,(factory,cols) in configs.items():
            d=datasets[name]
            parts=[]
            for fold in walk_forward_split(d,cfg["train_months"],cfg["embargo_hours"],cfg["first_test"],cfg["end_test_exclusive"]):
                tr=d.loc[fold['train']];te=d.loc[fold['test']]
                model=factory();model.fit(tr[cols],tr.target.astype(int))
                p=model.predict_proba(te[cols])[:,1]
                train_p=model.predict_proba(tr[cols])[:,1]
                all_folds.append({'asset':asset,'model':name,**{k:v for k,v in fold.items() if k not in ['train','test']},'train_n':len(tr),'test_n':len(te),**classification_metrics(te.target,p),'train_log_loss':classification_metrics(tr.target,train_p)['log_loss']})
                part=te[['timestamp','signal_time','entry_time','label_end_time','entry_price','exit_price','future_return','target']].copy()
                part['probability']=p;part['fold']=fold['fold'];part['model']=name;part['asset']=asset;parts.append(part)
                if name in ['logreg_funding','xgb_funding']:
                    values=model[-1].coef_[0] if name.startswith('logreg') else model.feature_importances_
                    for col,value in zip(cols,values):importance.append({'asset':asset,'model':name,'fold':fold['fold'],'feature':col,'value':float(value)})
            combined=pd.concat(parts,ignore_index=True)
            assert not combined.signal_time.duplicated().any()
            predictions.append(combined)
            summary[asset+'_'+name]=classification_metrics(combined.target,combined.probability)
            print(asset,name,summary[asset+'_'+name],flush=True)
        # Simple chronological baseline check, fixed 2025-09 cutoff, not independent of WF.
        d=datasets['logreg_funding']
        cutoff=pd.Timestamp(cfg['simple_holdout_cutoff'],tz='UTC').value//1_000_000
        tr=d[(d.signal_time<cutoff)&(d.label_end_time<cutoff-cfg["embargo_hours"]*3_600_000)]
        te=d[d.signal_time>=cutoff]
        for name,cols in [('logreg_price',PRICE_COLUMNS),('logreg_funding',FEATURE_COLUMNS)]:
            model=baseline_logreg();model.fit(tr[cols],tr.target.astype(int))
            holdout.append({'asset':asset,'model':name,'train_n':len(tr),**classification_metrics(te.target,model.predict_proba(te[cols])[:,1])})
    pd.DataFrame(all_folds).to_csv(OUT/'fold_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'oos_predictions.parquet',index=False)
    pd.DataFrame(importance).to_csv(OUT/'feature_importance.csv',index=False)
    pd.DataFrame(holdout).to_csv(OUT/'simple_holdout.csv',index=False)
    (OUT/'classification_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    # Paired month-block bootstrap: overlapping labels prevent independent-observation CI.
    folds=pd.DataFrame(all_folds);rng=np.random.default_rng(42);comparison=[]
    for asset in cfg['assets']:
        pivot=folds[folds.asset==asset].pivot(index='fold',columns='model',values='log_loss')
        for base,full in [('logreg_price','logreg_funding'),('xgb_price','xgb_funding')]:
            delta=(pivot[full]-pivot[base]).to_numpy()
            boot=rng.choice(delta,size=(3000,len(delta)),replace=True).mean(axis=1)
            comparison.append({'asset':asset,'comparison':full+' minus '+base,'mean_monthly_logloss_delta':float(delta.mean()),'month_block_bootstrap_95pct':[float(x) for x in np.quantile(boot,[.025,.975])]})
    (OUT/'funding_incremental_value.json').write_text(json.dumps(comparison,indent=2))


if __name__=='__main__':main()
