# %%
import threadpoolctl

threadpoolctl.threadpool_limits(limits=2)

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
        count.shift(4).rename("count_lagged_4h"),
        count.shift(24).rename("count_lagged_1d"),
        count.shift(24 + 1).rename("count_lagged_1d_1h"),
        count.shift(24 + 2).rename("count_lagged_1d_2h"),
        count.shift(24 + 3).rename("count_lagged_1d_3h"),
        count.shift(2 * 24 + 1).rename("count_lagged_2d_1h"),
        count.shift(2 * 24 + 2).rename("count_lagged_2d_2h"),
        count.shift(2 * 24 + 3).rename("count_lagged_2d_3h"),
        count.shift(7 * 24).rename("count_lagged_7d"),
        count.shift(7 * 24 + 1).rename("count_lagged_7d_1h"),
        count.shift(7 * 24 + 2).rename("count_lagged_7d_2h"),
        count.shift(7 * 24 + 3).rename("count_lagged_7d_3h"),
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
lagged_df = pd.concat(
    [count]
    + [count.shift(i).rename(f"count_lagged_{i}h") for i in range(1, 72)]
    + [count.shift(7 * 24 + i).rename(f"count_lagged_7d_{i}h") for i in range(0, 72)]
    + [
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
X.shape

# %%
from sklearn.model_selection import TimeSeriesSplit

ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    max_train_size=10000,
    test_size=1000,
)
# %%
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error


def evaluate(model, X, y, cv):
    def score_func(estimator, X, y):
        y_pred = estimator.predict(X)
        return {
            "mean_absolute_percentage_error": mean_absolute_percentage_error(y, y_pred),
            "root_mean_squared_error": np.sqrt(mean_squared_error(y, y_pred)),
            "mean_absolute_error": mean_absolute_error(y, y_pred),
            "mean_pinball_05_loss": mean_pinball_loss(y, y_pred, alpha=0.05),
            "mean_pinball_50_loss": mean_pinball_loss(y, y_pred, alpha=0.50),
            "mean_pinball_95_loss": mean_pinball_loss(y, y_pred, alpha=0.95),
        }

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=score_func,
    )
    for key, value in cv_results.items():
        if key.startswith("test_"):
            print(f"{key[5:]}: {value.mean():.3f} ± {value.std():.3f}")


# %%
from sklearn.ensemble import HistGradientBoostingRegressor

# %%
gbrt_mse = HistGradientBoostingRegressor(loss="squared_error")
evaluate(gbrt_mse, X, y, cv=ts_cv)

# %%
gbrt_poisson = HistGradientBoostingRegressor(loss="poisson")
evaluate(gbrt_poisson, X, y, cv=ts_cv)

# %%
gbrt_median = HistGradientBoostingRegressor(loss="quantile", quantile=0.05)
evaluate(gbrt_median, X, y, cv=ts_cv)

# %%
gbrt_median = HistGradientBoostingRegressor(loss="quantile", quantile=0.5)
evaluate(gbrt_median, X, y, cv=ts_cv)

# %%
gbrt_median = HistGradientBoostingRegressor(loss="quantile", quantile=0.95)
evaluate(gbrt_median, X, y, cv=ts_cv)

# %%
all_splits = list(ts_cv.split(X, y))
train_0, test_0 = all_splits[0]

# %%
gbrt_mean_poisson = HistGradientBoostingRegressor(loss="poisson")
gbrt_mean_poisson.fit(X.iloc[train_0], y.iloc[train_0])
mean_predictions = gbrt_mean_poisson.predict(X.iloc[test_0])

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
    "^-",
    label="GBRT median",
)
ax.plot(
    mean_predictions[last_hours],
    "x-",
    label="GBRT mean (Poisson)",
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
    ax.scatter(pred, y.iloc[test_0].values, alpha=0.3, label=label)
    ax.plot([0, y.max()], [0, y.max()], "--", label="Perfect model")
    ax.set(
        xlim=(0, y.max()),
        ylim=(0, y.max()),
        xlabel="Predicted demand",
        ylabel="True demand",
    )
    ax.legend()

plt.show()

# %%
(median_predictions > y.iloc[test_0]).mean()

# %%
(percentile_5_predictions > y.iloc[test_0]).mean()

# %%
(percentile_95_predictions > y.iloc[test_0]).mean()

# %%
np.logical_and(
    percentile_5_predictions < y.iloc[test_0],
    percentile_95_predictions > y.iloc[test_0],
).mean()

# %%
lagged_df = pd.concat(
    [
        count,
        count.shift(1).rename("count_lagged_1h"),
        count.shift(2).rename("count_lagged_2h"),
        count.shift(3).rename("count_lagged_3h"),
        count.shift(4).rename("count_lagged_4h"),
        count.shift(23).rename("count_lagged_23h"),
        count.shift(24).rename("count_lagged_1d"),
        count.shift(24 + 1).rename("count_lagged_1d_1h"),
        count.shift(24 + 2).rename("count_lagged_1d_2h"),
        count.shift(24 + 3).rename("count_lagged_1d_3h"),
        count.shift(2 * 24 + 1).rename("count_lagged_2d_1h"),
        count.shift(2 * 24 + 2).rename("count_lagged_2d_2h"),
        count.shift(2 * 24 + 3).rename("count_lagged_2d_3h"),
        count.shift(7 * 24).rename("count_lagged_7d"),
        count.shift(7 * 24 + 1).rename("count_lagged_7d_1h"),
        count.shift(7 * 24 + 2).rename("count_lagged_7d_2h"),
        count.shift(7 * 24 + 3).rename("count_lagged_7d_3h"),
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
lagged_df = pd.concat(
    [count]
    + [count.shift(i).rename(f"count_lagged_{i}h") for i in range(1, 72)]
    + [count.shift(7 * 24 + i).rename(f"count_lagged_7d_{i}h") for i in range(0, 72)]
    + [
        count.shift(1).rolling(24).mean().rename("lagged_mean_24h"),
        count.shift(1).rolling(24).max().rename("lagged_max_24h"),
        count.shift(1).rolling(24).min().rename("lagged_min_24h"),
        count.shift(1).rolling(7 * 24).mean().rename("lagged_mean_7d"),
        count.shift(1).rolling(7 * 24).max().rename("lagged_max_7d"),
        count.shift(1).rolling(7 * 24).min().rename("lagged_min_7d"),
    ],
    axis="columns",
).dropna()

X = lagged_df.drop("count", axis="columns")
y = lagged_df["count"]
all_splits = list(ts_cv.split(X, y))
train_0, test_0 = all_splits[0]

gbrt_mse = HistGradientBoostingRegressor(loss="squared_error")
gbrt_mse.fit(X.iloc[train_0], y.iloc[train_0])

gbrt_mean_poisson = HistGradientBoostingRegressor(loss="poisson")
gbrt_mean_poisson.fit(X.iloc[train_0], y.iloc[train_0])

gbrt_median = HistGradientBoostingRegressor(loss="quantile", quantile=0.5)
gbrt_median.fit(X.iloc[train_0], y.iloc[train_0])

gbrt_percentile_05 = HistGradientBoostingRegressor(loss="quantile", quantile=0.05)
gbrt_percentile_05.fit(X.iloc[train_0], y.iloc[train_0])

gbrt_percentile_95 = HistGradientBoostingRegressor(loss="quantile", quantile=0.95)
gbrt_percentile_95.fit(X.iloc[train_0], y.iloc[train_0])


# %%
from sklearn.inspection import permutation_importance
from sklearn.metrics import check_scoring


def boxplot_importances(model, X, y, scoring="r2", **kwargs):
    pi_results = permutation_importance(model, X, y, scoring=scoring, **kwargs)

    sorted_importances_idx = pi_results.importances_mean.argsort()[-15:]
    importances = pd.DataFrame(
        pi_results.importances[sorted_importances_idx].T,
        columns=X.columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances")
    ax.axvline(x=0, color="k", linestyle="--")
    ref_score = check_scoring(model, scoring)(model, X, y)
    ax.set_xlabel(f"Decrease in score (reference: {ref_score:.3f})")
    ax.figure.tight_layout()


# %%
boxplot_importances(gbrt_mse, X.iloc[test_0], y.iloc[test_0], scoring="r2")

# %%
boxplot_importances(gbrt_mean_poisson, X.iloc[test_0], y.iloc[test_0], scoring="r2")

# %%
from sklearn.metrics import d2_tweedie_score
from functools import partial
from sklearn.metrics import make_scorer

d2_poisson_score = make_scorer(partial(d2_tweedie_score, power=1))
boxplot_importances(
    gbrt_mean_poisson, X.iloc[test_0], y.iloc[test_0], scoring=d2_poisson_score
)

# %%
from sklearn.metrics import d2_pinball_score

d2_pinball_scorer_05 = make_scorer(partial(d2_pinball_score, alpha=0.05))
boxplot_importances(
    gbrt_percentile_05, X.iloc[test_0], y.iloc[test_0], scoring=d2_pinball_scorer_05
)

# %%
d2_pinball_scorer_95 = make_scorer(partial(d2_pinball_score, alpha=0.95))
boxplot_importances(
    gbrt_percentile_95, X.iloc[test_0], y.iloc[test_0], scoring=d2_pinball_scorer_95
)

# %%
