"""Download official archives or rebuild Parquet entirely offline."""
import argparse
from pathlib import Path
import pandas as pd
import yaml
from data.common import milliseconds
from data.archive import run_archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['data','features','train','report','all'], default='data')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--offline', action='store_true', help='Use local ZIP and CHECKSUM files; no network requests')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    start = milliseconds(args.start or config['start'])
    end = milliseconds(args.end or config['end'] or pd.Timestamp.now(tz='UTC'))
    if start >= end or not 1 <= args.workers <= 12:
        raise ValueError('Invalid time range or worker count (1–12)')
    if args.stage in ['data','all']:
        run_archive(config, start, end, args.offline, args.workers)
    if args.stage in ['features','all']:
        from data.audit_sol_gaps import main as repair
        repair(offline=args.offline)
        from features.build_dataset import main as build
        build()
    if args.stage in ['train','all']:
        from models.train_walkforward import main as train
        train()
    if args.stage in ['report','all']:
        from backtest.report import main as report
        from evaluation.robustness_check import main as robustness
        report()
        robustness()


if __name__ == '__main__':
    main()
