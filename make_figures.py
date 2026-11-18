from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight_rate.data import TARGET, engineer, read_loads
from freight_rate.validation import TEMPORAL_FOLDS, final_model, temporal_split

OUT = Path("outputs/figures")
TEAL = "#064A56"
ORANGE = "#D9822B"
GREY = "#8A9BA0"


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E1E8EA", linewidth=0.8)


def data_issues(train_raw, validation_raw):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.2), dpi=170)

    weight = train_raw["weight"].dropna()
    axes[0, 0].hist(weight, bins=120, color=TEAL)
    axes[0, 0].set_title("Weight (lb): sign flips and a hard cap", loc="left", fontsize=11, fontweight="bold")
    axes[0, 0].text(-47000, 1100, "negative values (about 0.6%)\nare sign flips: use |weight|", color=ORANGE, fontsize=9)
    axes[0, 0].text(-47000, 1600, "about 1,200 loads sit exactly at 47,500\n(a hard cap, not a real weight)", color=ORANGE, fontsize=9)
    axes[0, 0].set_xlabel("weight")

    rpm = train_raw[TARGET] / train_raw["distance"]
    axes[0, 1].hist(np.log10(rpm), bins=140, color=TEAL)
    axes[0, 1].set_title("Rate per mile: heavy two-sided tails", loc="left", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("log10($ per mile)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].annotate("about 1.4% of loads priced\nroughly 3-5x too high or too low", xy=(0.9, 20), xytext=(-0.45, 800), color=ORANGE, fontsize=9)

    for frame, label, color in ((train_raw, "train (Jan-Oct)", TEAL), (validation_raw, "validation (Nov-Dec)", ORANGE)):
        daily = frame.groupby("date")["market_index"].mean()
        axes[1, 0].plot(daily.index, daily.values, color=color, linewidth=1.4, label=label)
    axes[1, 0].set_title("Daily market_index: level shifts in the validation period", loc="left", fontsize=11, fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].tick_params(axis="x", rotation=30)

    missing = pd.DataFrame(
        {
            "train": train_raw[["weight", "market_index"]].isna().mean() * 100,
            "validation": validation_raw[["weight", "market_index"]].isna().mean() * 100,
        }
    )
    positions = np.arange(len(missing))
    axes[1, 1].bar(positions - 0.18, missing["train"], 0.36, color=TEAL, label="train")
    axes[1, 1].bar(positions + 0.18, missing["validation"], 0.36, color=ORANGE, label="validation")
    axes[1, 1].set_xticks(positions, missing.index)
    axes[1, 1].set_ylabel("% missing")
    axes[1, 1].set_title("Missing values", loc="left", fontsize=11, fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=9)

    for axis in axes.ravel():
        style(axis)
    figure.tight_layout()
    figure.savefig(OUT / "data_issues.png")
    plt.close(figure)


def split_design(frame):
    figure, axis = plt.subplots(figsize=(11, 2.9), dpi=170)
    start = pd.Timestamp("2025-01-01")
    spans = [
        ("Fold 1", "2025-01-01", "2025-04-30", "2025-05-01", "2025-06-30"),
        ("Fold 2", "2025-01-01", "2025-06-30", "2025-07-01", "2025-08-31"),
        ("Fold 3 (primary)", "2025-01-01", "2025-08-31", "2025-09-01", "2025-10-31"),
    ]
    for row, (name, a, b, c, d) in enumerate(spans[::-1]):
        axis.barh(row, (pd.Timestamp(b) - pd.Timestamp(a)).days, left=(pd.Timestamp(a) - start).days, color=TEAL, height=0.5)
        axis.barh(row, (pd.Timestamp(d) - pd.Timestamp(c)).days, left=(pd.Timestamp(c) - start).days, color=ORANGE, height=0.5)
    axis.barh(3, 61, left=(pd.Timestamp("2025-11-01") - start).days, color=GREY, height=0.5)
    axis.barh(3, 304, left=0, color=TEAL, height=0.5, alpha=0.35)
    axis.set_yticks(range(4), [s[0] for s in spans[::-1]] + ["Final: train all, predict"])
    months = pd.date_range("2025-01-01", "2026-01-01", freq="MS")
    axis.set_xticks([(m - start).days for m in months], [m.strftime("%b") for m in months])
    axis.set_title("Forward-chaining validation: always train on the past, test on the next two months", loc="left", fontsize=11, fontweight="bold")
    figure.text(0.01, 0.015, "teal = training window    orange = test window    grey = Nov-Dec 2025 prediction period (no labels)", fontsize=9, color="#455A60")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(OUT / "split_design.png")
    plt.close(figure)


def holdout_fit(frame):
    train_end, test_start, test_end = TEMPORAL_FOLDS[-1][1:]
    train, test = temporal_split(frame, train_end, test_start, test_end)
    model = final_model().fit(train)
    predicted = model.predict(test)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=170)
    clean = np.abs(np.log(test[TARGET] / predicted)) < 0.4
    axes[0].scatter(test[TARGET][clean], predicted[clean], s=3, color=TEAL, alpha=0.35, label="clean loads")
    axes[0].scatter(test[TARGET][~clean], predicted[~clean], s=14, color=ORANGE, label="outlier-priced loads")
    limit = float(test[TARGET].quantile(0.999))
    axes[0].plot([0, limit], [0, limit], color="#455A60", linewidth=1)
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("actual posted rate ($)")
    axes[0].set_ylabel("predicted ($)")
    axes[0].set_title("Held-out Sep-Oct 2025", loc="left", fontsize=11, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    error = (np.log(test[TARGET]) - np.log(predicted))[clean]
    daily = error.groupby(test.loc[error.index, "date"]).mean() * 100
    axes[1].bar(daily.index, daily.values, color=TEAL, width=0.9)
    axes[1].axhline(0, color="#455A60", linewidth=1)
    axes[1].set_ylabel("mean error (%, positive = under-predicted)")
    axes[1].set_title("Average daily error on the held-out window", loc="left", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=30)
    for axis in axes:
        style(axis)
    figure.tight_layout()
    figure.savefig(OUT / "holdout_fit.png")
    plt.close(figure)


def comparison():
    table = pd.read_csv("outputs/validation_temporal_folds.csv")
    mean = table.groupby("model")["clean_mape_pct"].mean().sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(11, 4.6), dpi=170)
    colors = [ORANGE if name.startswith("FINAL") else TEAL for name in mean.index]
    axis.barh(range(len(mean)), mean.values, color=colors)
    axis.set_yticks(range(len(mean)), [n if len(n) < 58 else n[:55] + "..." for n in mean.index], fontsize=8.5)
    for position, value in enumerate(mean.values):
        axis.text(value + 0.08, position, f"{value:.2f}%", va="center", fontsize=8.5)
    axis.set_xlabel("MAPE on clean loads, averaged over the three forward-chaining folds (lower is better)")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUT / "model_comparison.png")
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_raw = read_loads("data/train_test.csv")
    validation_raw = read_loads("data/validation.csv")
    frame = engineer(train_raw)
    data_issues(train_raw, validation_raw)
    split_design(frame)
    holdout_fit(frame)
    comparison()


if __name__ == "__main__":
    main()
