# Loom talking points (aim for 2:30, camera on)

Read these as bullets, not a script. Rough timings in brackets. Have `outputs/figures/data_issues.png`, the report, and `freight_rate/model.py` open in tabs.

## 1. Key findings from exploring the data [0:00-0:40]
- 48k training loads from Jan to Oct, 12k validation loads from Nov and Dec. The unlabelled period is entirely in the future, so this is really a forecasting problem.
- Rate is mostly a function of distance, equipment and weight, then two time-varying things: a daily market_index and a slow upward drift of about 6% a year.
- Geography matters: northern lanes are cheaper than Texas and Oklahoma lanes. 8 of the 72 validation cities never appear in training.
- quote_signal has a small U-shaped effect; its daily average moves in training but is flat in validation, so I only use it per load.

## 2. Data-quality issues and fixes [0:40-1:15]
- About 1.4% of rates are contaminated: roughly 3 to 5 times too high or too low. I drop them from fitting with an iterative residual rule. Doing this cuts the error by about 48%.
- Negative weights are sign flips, so I use the absolute value. 1,200 weights sit at exactly 47,500, a cap, so I flag them.
- Missing market_index is filled with that day's average; weight is left missing and the boosting model handles it.

## 3. Training and validation approach, and the split [1:15-1:50]
- Time-based, forward-chaining: train on the past, test on the next two months. Three folds; the last (train Jan to Aug, test Sep to Oct) is the one that matches the real task.
- A random split flatters the model: a tree model that sees dates reports about 42% lower error under a random split than it really has.
- Also tested held-out cities to check that coordinates generalise to cities not seen in training.
- Result on the primary fold: MAE $88.3, median error 1.15%, MAPE 1.39% on regular loads. Naive baseline: MAPE 10.5%.

## 4. Reasoning behind the model [1:50-2:15]
- log(rate) = load part + market part. Boosting for the load part; a linear market term plus drift for the market part.
- Why: trees cannot extrapolate a trend and memorise dates. The structured model scored 1.87% versus 4.25% for trees given market_index and a day index.
- The market coefficient is estimated from within-day variation so it is not confused with seasonality.

## 5. Quick code walkthrough [2:15-2:30]
- `freight_rate/data.py`: cleaning and features.
- `freight_rate/model.py`: the alternating fit in `fit()` (boosting round, then day-term regression, then outlier mask).
- `run_validation.py` and `train_predict.py`: validation and final predictions. Then show the December chart.

## Be ready to answer
- Why linear drift rather than holding it flat? Both score about the same in validation (1.87% vs 1.86%); the difference is at most about 1.0% by end of December.
- What is the biggest weakness? Only 10 months of history, so month-level seasonality for Nov and Dec is unknown.
