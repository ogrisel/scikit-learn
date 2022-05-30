# %%
# Data exploration on the Bike Sharing Demand dataset
# ---------------------------------------------------
#
# We start by loading the data from the OpenML repository.
from sklearn.datasets import fetch_openml

bike_sharing = fetch_openml(
    "Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas"
)
df = bike_sharing.frame
count = df["count"]  # / df["count"].max()

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 4))
count.hist(bins=30, ax=ax)
_ = ax.set(
    xlabel="Fraction of rented fleet demand",
    ylabel="Number of hours",
)

# %%
import pandas as pd

lagged_df = pd.concat(
    [
        count,
        count.shift(1).rename("count_lagged_1h"),
        count.shift(2).rename("count_lagged_2h"),
        count.shift(3).rename("count_lagged_3h"),
        count.shift(1).rolling(24).mean().rename("lagged_mean_24h"),
        count.shift(1).rolling(24).max().rename("lagged_max_24h"),
        count.shift(1).rolling(24).min().rename("lagged_min_24h"),
        count.shift(1).rolling(7 * 24).mean().rename("lagged_mean_7d"),
        count.shift(1).rolling(7 * 24).max().rename("lagged_max_7d"),
        count.shift(1).rolling(7 * 24).min().rename("lagged_min_7d"),
    ],
    axis="columns",
).dropna()


# %%
X = lagged_df.drop("count", axis="columns")
y = lagged_df["count"]
# %%
from sklearn.model_selection import TimeSeriesSplit

ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    max_train_size=10000,
    test_size=1000,
)
# %%
from sklearn.model_selection import cross_validate


def evaluate(model, X, y, cv):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
            "neg_mean_absolute_percentage_error",
        ],
    )
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    mape = -cv_results["test_neg_mean_absolute_percentage_error"] * 100
    print(
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}\n"
        f"MAPE:                    {mape.mean():.1f} +/- {mape.std():.1f}%"
    )


# %%
from sklearn.ensemble import HistGradientBoostingRegressor

# %%
gbrt_mse = HistGradientBoostingRegressor(loss="squared_error")
evaluate(gbrt_mse, X, y, cv=ts_cv)

# %%
gbrt_poisson = HistGradientBoostingRegressor(loss="poisson")
evaluate(gbrt_poisson, X, y, cv=ts_cv)

# %%
gbrt_median = HistGradientBoostingRegressor(loss="quantile", quantile=0.5)
evaluate(gbrt_median, X, y, cv=ts_cv)
# %%
all_splits = list(ts_cv.split(X, y))
train_0, test_0 = all_splits[0]

# %%
gbrt_median = HistGradientBoostingRegressor(loss="quantile", quantile=0.5)
gbrt_median.fit(X.iloc[train_0], y.iloc[train_0])
median_predictions = gbrt_median.predict(X.iloc[test_0])

gbrt_percentile_5 = HistGradientBoostingRegressor(loss="quantile", quantile=0.05)
gbrt_percentile_5.fit(X.iloc[train_0], y.iloc[train_0])
percentile_5_predictions = gbrt_percentile_5.predict(X.iloc[test_0])

gbrt_percentile_95 = HistGradientBoostingRegressor(loss="quantile", quantile=0.95)
gbrt_percentile_95.fit(X.iloc[train_0], y.iloc[train_0])
percentile_95_predictions = gbrt_percentile_95.predict(X.iloc[test_0])


# %%
import numpy as np

last_hours = slice(-96, None)
fig, ax = plt.subplots(figsize=(12, 4))
fig.suptitle("Predictions by regression models")
ax.plot(
    y.iloc[test_0].values[last_hours],
    "x-",
    alpha=0.2,
    label="Actual demand",
    color="black",
)
ax.plot(
    median_predictions[last_hours],
    "x-",
    label="GBRT median",
)
ax.fill_between(
    np.arange(96),
    percentile_5_predictions[last_hours],
    percentile_95_predictions[last_hours],
    alpha=0.3,
)
_ = ax.legend()

# %%
fig, axes = plt.subplots(ncols=3, figsize=(12, 4), sharey=True)
fig.suptitle("Non-linear regression models")
predictions = [
    median_predictions,
    percentile_5_predictions,
    percentile_95_predictions,
]
labels = [
    "Median",
    "5th percentile",
    "95th percentile",
]
for ax, pred, label in zip(axes, predictions, labels):
    ax.scatter(y.iloc[test_0].values, pred, alpha=0.3, label=label)
    ax.plot([0, y.max()], [0, y.max()], "--", label="Perfect model")
    ax.set(
        xlim=(0, y.max()),
        ylim=(0, y.max()),
        xlabel="True demand",
        ylabel="Predicted demand",
    )
    ax.legend()

plt.show()

# %%
(percentile_5_predictions < y.iloc[test_0]).mean()

# %%
(percentile_95_predictions > y.iloc[test_0]).mean()

# %%
np.logical_and(
    percentile_5_predictions < y.iloc[test_0],
    percentile_95_predictions > y.iloc[test_0],
).mean()

# %%
