# Freight Rate Prediction Challenge

Solution for a Machine Learning Engineer take-home assessment (freight rate prediction). The model predicts the posted rate of every load in `data/validation.csv` (Nov-Dec 2025) from the labelled loads in `data/train_test.csv` (Jan-Oct 2025).

## Result in one paragraph

Forward-chaining validation (always train on the past, test on the next two months). On the primary fold (train Jan-Aug, test Sep-Oct): MAE $88.3, median error 1.15%, MAPE 1.39% on regular loads and 3.93% on all loads. A naive $/mile-by-equipment baseline gets MAE $229 and MAPE 10.5%. Full details, figures and reasoning are in `outputs/Freight_Rate_Report.docx`.

## Approach

- `log(rate)` is modelled as a load part plus a market part.
- Load part: histogram gradient boosting on log distance, equipment, absolute weight and weight flags, `quote_signal`, and coordinate-based geography (no city IDs, so unseen cities work).
- Market part: a linear term on the per-load `market_index` (coefficient estimated from within-day variation) plus a slow linear time drift.
- About 1.4% of training rates are contaminated outliers. They are removed from fitting with an iterative residual rule, and the two parts are fitted by alternating three rounds.
- Data fixes: negative weights are sign flips (absolute value used), weight capped at 47,500 gets a flag, missing `market_index` is filled with that day's average, missing weight is left to the boosting model.

## Setup

Python 3.10+.

```bash
python -m pip install -r requirements.txt
```

Put the provided files in `data/`:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Run

```bash
python run_validation.py
python train_predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python make_figures.py
```

- `run_validation.py` runs the three forward-chaining folds for all model variants, a random 5-fold contrast, and the unseen-city test. Results go to `outputs/validation_*.csv` and `outputs/validation_summary.json`. It takes a few minutes.
- `train_predict.py` fits the final model on all 48,000 labelled loads, writes `validation_predictions.csv` (columns `load_id,predicted_rate`), and fills `predicted_rate` in `data/december_chart_inputs.csv`. Fitted coefficients are saved to `outputs/final_model_details.json`.
- `score.py` is the provided scorer. It validates both files and creates `scorer_results/candidate_december.png`.
- `make_figures.py` regenerates the report figures in `outputs/figures/`.

## Layout

```
freight_rate/data.py         loading, cleaning and feature engineering
freight_rate/model.py        the hybrid boosting plus market model
freight_rate/metrics.py      error metrics (all loads and regular loads)
freight_rate/validation.py   forward-chaining folds, baselines, unseen-city test
freight_rate/predict.py      builds the December chart features
run_validation.py            validation experiments
train_predict.py             final fit, predictions, December predictions
make_figures.py              report figures
validation_predictions.csv   final predictions for the 12,000 validation loads
outputs/                     report, validation tables, figures, model details
```

## Notes and limits

- The December chart uses each date's average `market_index` from `validation.csv` as a known input, and the median `quote_signal` for the hypothetical load. It has no holiday information.
- "Regular loads" means loads whose rate is within a factor of about 1.5 of the prediction; the roughly 1.4% contaminated loads are included in the "all loads" figures.
- Everything uses fixed random seeds. The committed `validation_predictions.csv` and `scorer_results/candidate_december.png` were produced with Python 3.13, scikit-learn 1.9.1, pandas 2.3.3 and numpy 2.5.3. Re-running with the same versions reproduces them exactly; other scikit-learn versions give very slightly different boosting results (predictions typically differ by well under 1%, and the December chart by about 0.2%).
- Because of that, the report figures correspond to the committed output files rather than to a fresh re-run on a different library version.
