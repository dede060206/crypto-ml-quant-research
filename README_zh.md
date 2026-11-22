# Crypto ML Quant：资金费率方向预测研究

用于金融／量化方向研究生申请的研究作品集。研究问题：历史资金费率异常是否为BTC、ETH、SOL永续合约未来24小时方向提供额外预测信息？研究强调可信验证与可复现性，结果不需要表现为赚钱。

## 已完成范围与结论

README原始规划的八个开发阶段已完成：官方离线数据、Buy & Hold地基核对、历史特征与标签、逻辑回归基线、walk-forward切分、XGBoost样本外训练、三个资产／窗口／交易成本稳健性分析和研究报告。

2022-01至2026-08，每个资产56个测试月份，12个模型／特征版本，共2016条fold结果。完整离线入口已成功复跑；34项测试通过。六份QuantStats报告的日Sharpe、Sortino和日最大回撤与手写公式一致。

本实验没有支持稳定的资金费率增量预测优势，含成本主模型策略在三个资产上均亏损。保留所有版本与失败结果，不用样本外收益反向挑选参数。新增窗口与SOL实验属于收尾扩展探索，不能作为独立确认性证据。

## 先阅读这些文件

- `reports/ml_research/research_report.html`：完整可读研究报告，含表格、权益曲线和逐月AUC图。
- `notebooks/research_report.ipynb`：已执行、保存表格和图的Notebook。
- `reports/ml_research/fold_metrics.csv`、`oos_predictions.parquet`：全部折次结果与预测。
- `reports/ml_research/backtest_summary.json`：含／不含成本、逐年收益、回撤、交易指标和滑点敏感性。
- `reports/ml_research/run_metadata.json`：模型配置、输入SHA256与依赖版本。
- `reports/final_acceptance.json`：最终验收记录。

## 数据与缺口处理

[Binance官方历史文件站](https://data.binance.vision/)提供ZIP；每个文件逐一通过官方SHA256与ZIP校验。[官方格式说明](https://github.com/binance/binance-public-data/)

408个永续月度文件覆盖2021-01至2026-08；56个BTC现货月度文件覆盖样本外区间。SOL月度文件缺少15根8h K线，本次从官方每日8h文件补齐全部15根真实记录，相邻日期12条日／月重叠记录的OHLCV一致；没有插值或补零。原始月度数据、缺失时间、每日文件与校验值均保留。SOL资金结算间隔变化按实际记录处理，间隔异常边界重启资金统计。

原始ZIP在`data/raw_archive/`，Parquet与来源清单在`data/cache/`，特征与标签在`data/processed/`。当前各资产8h行情均6207根、无缺口；BTC/ETH各6182条可训练样本，SOL6180条。现货2025起微秒时间戳显式转换为毫秒。官方文件可能修订，本地校验值与输入哈希固定本次版本。

## 方法与限制

13项主特征仅使用已收盘价格和结算后延迟60秒视为可用的资金费率。资金统计在真实结算事件层计算；标准化器仅在每折训练集拟合。标签为下一根开盘入场、24小时后开盘退出的上涨是否超过预先固定约0.160128%双边执行成本阈值，模型输入使用显式特征白名单。

训练窗口12个日历月、测试1个月、前推1个月；按实际label_end_time剔除重叠训练标签，额外隔离8小时。不在外层测试集调参或early stopping。保留价格特征对照、Dummy基线、逻辑回归与XGBoost。除主24h/3d/7d窗口外，实际重算24h/2d/5d与48h/5d/14d，并在相同样本池重新训练全部折次。

概率>=0.55做多，否则空仓；每次10%权益预算，固定单位持有24小时、不重叠。手续费每侧0.05%，滑点每侧0.03%。资金费用用真实费率和结算前已收盘价格代理标记价格，因此净收益为研究性近似。真实BTC现货Buy & Hold提供100%和10%初始投入参考；不同风险预算不能只按绝对收益判优劣。未模拟完整保证金清算、订单延迟或盘口冲击；60秒公布延迟也是假设，不是观测到的历史公布时间。

详见`reports/DATASET_METHOD.md`、`reports/buy_hold_validation/VALIDATION.md`和`reports/ml_research/RESEARCH_REPORT.md`。原始需求保存在`RESEARCH_SPEC.md`。选择现存主流币仍有生存者偏差。已经看过的样本外结果不能用于严格确认后续新假设。

## 安装与离线重现

测试环境为Python 3.14、macOS ARM64。新环境安装：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

`requirements-lock.txt`固定当前环境依赖；跨平台使用项目依赖解析。macOS的XGBoost需要libomp。本机已在`work/runtime/`配置经过Homebrew官方校验的运行库；另一个Mac若未复制该库，可安装系统libomp。打包文件不含本机虚拟环境或本机运行库，安装方式参见[XGBoost官方说明](https://xgboost.readthedocs.io/en/stable/install.html)。

完整复跑（仅使用本地ZIP与CHECKSUM，无需交易API）：

```sh
./run_research.sh
```

固定截至2026-09-01 UTC，执行测试、原始数据导入、SOL每日来源补齐、特征构造、全部训练、交易回测和报告。六份QuantStats HTML、已执行Notebook和全部版本结果重新生成。

原始数据不在本地时先下载：

```sh
.venv/bin/python main.py --stage data --end 2026-09-01T00:00:00Z
.venv/bin/python -m data.audit_sol_gaps
.venv/bin/python -m data.spot_benchmark
```

单阶段入口：`main.py --stage features`、`--stage train`、`--stage report`；`--stage all`为整合入口。macOS单独训练须像run_research.sh一样设置DYLD_LIBRARY_PATH。配置在`configs/config.yaml`和`configs/training.yaml`。

修改参数、交易阈值或数据版本后应另存结果，并标明探索性质；本次作品集完成不等于策略适合实盘交易。
