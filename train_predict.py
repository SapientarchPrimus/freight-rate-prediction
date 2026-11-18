import json
from pathlib import Path

import numpy as np
import pandas as pd

from freight_rate.data import TARGET, city_table, engineer, read_loads
from freight_rate.predict import december_frame
from freight_rate.validation import final_model

OUT = Path("outputs")


def main():
    OUT.mkdir(exist_ok=True)
    train_raw = read_loads("data/train_test.csv")
    validation_raw = read_loads("data/validation.csv")
    template = pd.read_csv("data/validation_predictions_template.csv")
    chart_inputs = pd.read_csv("data/december_chart_inputs.csv")

    train = engineer(train_raw)
    validation = engineer(validation_raw)

    model = final_model().fit(train)
    predictions = model.predict(validation)

    result = pd.DataFrame({"load_id": validation_raw["load_id"], "predicted_rate": np.round(predictions, 2)})
    if set(result["load_id"]) != set(template["load_id"]) or len(result) != len(template):
        raise SystemExit("validation load_ids do not match the template")
    result = template[["load_id"]].merge(result, on="load_id", how="left")
    result.to_csv("validation_predictions.csv", index=False)

    quote_signal = float(pd.concat([train["quote_signal"], validation["quote_signal"]]).median())
    cities = city_table(train_raw, validation_raw)
    december = december_frame(chart_inputs, cities, validation_raw, quote_signal)
    filled = chart_inputs.copy()
    filled["predicted_rate"] = np.round(model.predict(december), 2)
    filled.to_csv("data/december_chart_inputs.csv", index=False)
    filled.to_csv(OUT / "december_chart_inputs_predicted.csv", index=False)

    details = {
        "market_coefficients": model.day_coefficients(),
        "intercept": model.intercept_,
        "training_rows": int(len(train)),
        "training_rows_used_after_outlier_filter": int(model.keep_.sum()),
        "outliers_dropped_from_fit": int((~model.keep_).sum()),
        "robust_residual_sigma_log": model.sigma_,
        "december_quote_signal_used": quote_signal,
        "prediction_summary": result["predicted_rate"].describe().round(2).to_dict(),
        "december_summary": filled["predicted_rate"].describe().round(2).to_dict(),
    }
    (OUT / "final_model_details.json").write_text(json.dumps(details, indent=2))
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
