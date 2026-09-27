"""CSCI 270 Group Project 1.

Each teammate's work should be placed under the appropriate part marker. This
file must run from top to bottom for the final submission.
"""


# === PART 1: CLASSIFICATION ===
# Teammate code goes here.


# === PART 2: REGRESSION ===
# Teammate code goes here.


# === PART 3: CLUSTERING ===
# Teammate code goes here.


# === PART 4: TIME SERIES - AIR QUALITY AND THE AI AUDIT ===

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATION = "Aotizhongxin"
POLLUTANT = "PM2.5"

# This is the exact baseline prompt used for the AI audit:
#
# "Write Python code that loads the Aotizhongxin station from the Beijing
# Multi-Site Air Quality Dataset, combines the year, month, day, and hour
# columns into a datetime index, fills missing PM2.5 values, resamples the
# series to daily averages, and fits a trend line."
#
# A typical minimal AI answer applies unrestricted linear interpolation. The
# baseline below implements that answer so it can be measured against our fix.


def find_station_csv() -> Path:
    """Return the Aotizhongxin CSV path from common project locations."""
    project_dir = Path(__file__).resolve().parent
    filename = "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
    candidates = (
        project_dir / filename,
        project_dir / "data" / filename,
        project_dir / "PRSA_Data_20130301-20170228" / filename,
        project_dir.parent / filename,
        project_dir.parent / "data" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find {filename}. Searched:\n  - {searched}\n"
        "Download the Beijing Multi-Site Air Quality Dataset from UCI and "
        "place this station CSV in Group-Project-1/data/."
    )


def load_hourly_pm25(csv_path: Path) -> pd.Series:
    """Load one station and create a complete, validated hourly PM2.5 series."""
    station_data = pd.read_csv(csv_path, na_values=["NA"])
    required = {"year", "month", "day", "hour", POLLUTANT}
    missing_columns = required.difference(station_data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    # Enforce the exactly-one-station requirement even if a combined CSV is used.
    if "station" in station_data.columns:
        station_data = station_data.loc[station_data["station"] == STATION].copy()
        if station_data.empty:
            raise ValueError(f"No rows were found for station {STATION!r}.")

    station_data["datetime"] = pd.to_datetime(
        station_data[["year", "month", "day", "hour"]], errors="raise"
    )
    if station_data["datetime"].duplicated().any():
        raise ValueError("Duplicate hourly timestamps were found in the station data.")

    hourly = (
        station_data.set_index("datetime")[POLLUTANT]
        .astype(float)
        .sort_index()
        .rename(POLLUTANT)
    )

    # Reindexing exposes missing timestamps as NaN instead of silently skipping them.
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    return hourly.reindex(full_index).rename_axis("datetime")


def missing_run_lengths(series: pd.Series) -> pd.Series:
    """Return the length of every consecutive run of missing observations."""
    is_missing = series.isna()
    run_id = is_missing.ne(is_missing.shift(fill_value=False)).cumsum()
    return is_missing.groupby(run_id).sum().loc[lambda lengths: lengths > 0]


def print_missingness_audit(hourly: pd.Series) -> None:
    """Print evidence used to choose and evaluate the imputation strategy."""
    gaps = missing_run_lengths(hourly)
    total = len(hourly)
    missing = int(hourly.isna().sum())

    print("\n=== PART 4: MISSING-DATA AUDIT ===")
    print(f"Station: {STATION}")
    print(f"Pollutant: {POLLUTANT}")
    print(f"Hourly observations expected: {total:,}")
    print(f"Missing hourly values: {missing:,} ({missing / total:.2%})")
    print(f"Consecutive missing segments: {len(gaps):,}")
    print(f"Gaps of 6 hours or less: {int((gaps <= 6).sum()):,}")
    print(f"Gaps longer than 6 hours: {int((gaps > 6).sum()):,}")
    print(f"Longest missing segment: {int(gaps.max()) if len(gaps) else 0:,} hours")


def ai_baseline_imputation(hourly: pd.Series) -> pd.Series:
    """Reproduce the initial AI baseline: unrestricted time interpolation."""
    # This can draw an artificial straight line across a multi-day sensor outage.
    return hourly.interpolate(method="time", limit_direction="both")


def audited_gap_aware_imputation(
    hourly: pd.Series, short_gap_hours: int = 6
) -> pd.Series:
    """Fill short gaps locally and long gaps with seasonal, time-of-day medians."""
    if short_gap_hours < 1:
        raise ValueError("short_gap_hours must be at least 1.")

    original_missing = hourly.isna()
    is_short_gap = pd.Series(False, index=hourly.index)

    # The audit found that 196 of 209 gaps are at most six hours, making six
    # hours a data-supported definition of a routine short outage. Mark entire
    # runs so that no part of a long gap is accidentally interpolated.
    is_missing = hourly.isna()
    run_id = is_missing.ne(is_missing.shift(fill_value=False)).cumsum()
    for gap_id, length in is_missing.groupby(run_id).sum().items():
        if 0 < length <= short_gap_hours:
            is_short_gap.loc[run_id == gap_id] = True

    local_interpolation = hourly.interpolate(method="time", limit_area="inside")
    repaired = hourly.copy()
    repaired.loc[is_short_gap] = local_interpolation.loc[is_short_gap]

    # Long gaps use only observed values from the same month and hour of day,
    # preserving broad seasonal and daily cycles without inventing a long line.
    observed = hourly.loc[~original_missing]
    seasonal_medians = observed.groupby(
        [observed.index.month, observed.index.hour]
    ).median()
    for timestamp in repaired.index[repaired.isna()]:
        repaired.loc[timestamp] = seasonal_medians.get(
            (timestamp.month, timestamp.hour), np.nan
        )

    # Defensive fallback for a month/hour group with no observed values at all.
    if repaired.isna().any():
        monthly_medians = observed.groupby(observed.index.month).median()
        for timestamp in repaired.index[repaired.isna()]:
            repaired.loc[timestamp] = monthly_medians.get(timestamp.month, np.nan)
    if repaired.isna().any():
        repaired = repaired.fillna(observed.median())

    if repaired.isna().any():
        raise RuntimeError("Imputation unexpectedly left missing PM2.5 values.")
    return repaired


def daily_average_and_trend(
    hourly: pd.Series,
) -> tuple[pd.Series, np.ndarray, float, float]:
    """Resample to daily means and fit an ordinary least-squares linear trend."""
    daily = hourly.resample("D").mean()
    elapsed_days = (daily.index - daily.index[0]).days.to_numpy(dtype=float)
    slope_per_day, intercept = np.polyfit(elapsed_days, daily.to_numpy(), deg=1)
    fitted = intercept + slope_per_day * elapsed_days

    residual_sum_squares = np.sum((daily.to_numpy() - fitted) ** 2)
    total_sum_squares = np.sum((daily.to_numpy() - daily.mean()) ** 2)
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    return daily, fitted, slope_per_day * 365.25, r_squared


def plot_daily_trend(
    daily: pd.Series, fitted: np.ndarray, slope_per_year: float, r_squared: float
) -> None:
    """Plot daily PM2.5 observations and the overall fitted trend line."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.scatter(
        daily.index,
        daily,
        s=9,
        alpha=0.35,
        color="steelblue",
        label="Daily average PM2.5",
    )
    axis.plot(
        daily.index,
        fitted,
        linewidth=2.5,
        color="darkred",
        label="Linear trend",
    )
    axis.set_title(f"Daily PM2.5 and Overall Trend — {STATION} Station")
    axis.set_xlabel("Date")
    axis.set_ylabel("PM2.5 concentration (µg/m³)")
    axis.grid(alpha=0.2)
    axis.legend()
    axis.text(
        0.02,
        0.96,
        f"Slope: {slope_per_year:.2f} µg/m³ per year\n$R^2$: {r_squared:.3f}",
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    figure.tight_layout()
    plt.show()


def run_part_4() -> None:
    """Run the baseline, audit it, apply the correction, and make the final plot."""
    csv_path = find_station_csv()
    hourly = load_hourly_pm25(csv_path)
    print_missingness_audit(hourly)

    baseline = ai_baseline_imputation(hourly)
    corrected = audited_gap_aware_imputation(hourly, short_gap_hours=6)
    originally_missing = hourly.isna()
    mean_difference = (
        baseline[originally_missing] - corrected[originally_missing]
    ).abs().mean()

    print("\n=== BASELINE VS. AUDITED METHOD ===")
    print("AI baseline: unrestricted time interpolation")
    print("Audited method: <=6-hour interpolation; longer gaps use month-hour medians")
    print(f"Mean absolute difference at missing points: {mean_difference:.2f} µg/m³")

    daily, fitted, slope_per_year, r_squared = daily_average_and_trend(corrected)
    print("\n=== FINAL DAILY TREND ===")
    print(f"Daily observations: {len(daily):,}")
    print(f"Trend slope: {slope_per_year:.2f} µg/m³ per year")
    print(f"Trend R²: {r_squared:.3f}")
    plot_daily_trend(daily, fitted, slope_per_year, r_squared)


if __name__ == "__main__":
    run_part_4()
