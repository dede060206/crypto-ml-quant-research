"""Summarize all fixed sensitivity variants; never choose a winner."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports'/'ml_research'
    folds=pd.read_csv(out/'fold_metrics.csv')
    rows=[]
    for (asset,model),part in folds.groupby(['asset','model']):
        rows.append({'asset':asset,'model':model,'folds':len(part),'fold_auc_mean':float(part.roc_auc.mean()),'fold_auc_std':float(part.roc_auc.std()),'fold_auc_min':float(part.roc_auc.min()),'fold_auc_max':float(part.roc_auc.max()),'fraction_folds_auc_above_05':float((part.roc_auc>.5).mean()),'mean_train_logloss':float(part.train_log_loss.mean()),'mean_test_logloss':float(part.log_loss.mean())})
    pd.DataFrame(rows).to_csv(out/'robustness_summary.csv',index=False)
    print('All model variants retained; robustness_summary.csv generated')


if __name__=='__main__':main()
