import json
from pathlib import Path

import pandas as pd

from freight_rate.data import engineer, read_loads
from freight_rate.validation import PRIMARY_FOLD, model_zoo, run_random_kfold, run_temporal, run_unseen_cities

OUT = Path("outputs")


def main():
    OUT.mkdir(exist_ok=True)
    frame = engineer(read_loads("data/train_test.csv"))

    temporal = run_temporal(frame, model_zoo())
    temporal.to_csv(OUT / "validation_temporal_folds.csv", index=False)

    kfold = run_random_kfold(frame)
    kfold.to_csv(OUT / "validation_random_kfold.csv", index=False)

    unseen = run_unseen_cities(frame)
    unseen.to_csv(OUT / "validation_unseen_cities.csv", index=False)

    columns = ["mae", "rmse", "mape_pct", "median_ape_pct", "r2", "clean_mape_pct"]
    summary = temporal.groupby("model")[columns].mean().round(3)
    primary = temporal[temporal["fold"] == PRIMARY_FOLD].set_index("model")[columns].round(3)
    random_mean = kfold.groupby("model")[columns].mean().round(3)
    unseen_summary = unseen.groupby(["model", "subset"])[["mape_pct", "clean_mape_pct"]].mean().round(3)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\nMean across the three forward-chaining folds")
    print(summary.sort_values("clean_mape_pct"))
    print("\nPrimary fold (train Jan-Aug, test Sep-Oct)")
    print(primary.sort_values("clean_mape_pct"))
    print("\nRandom 5-fold (optimistic, for contrast)")
    print(random_mean)
    print("\nUnseen cities")
    print(unseen_summary)

    payload = {
        "temporal_mean": summary.to_dict("index"),
        "primary_fold": primary.to_dict("index"),
        "random_kfold_mean": random_mean.to_dict("index"),
        "unseen_cities": {f"{a} | {b}": v for (a, b), v in unseen_summary.to_dict("index").items()},
    }
    (OUT / "validation_summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
