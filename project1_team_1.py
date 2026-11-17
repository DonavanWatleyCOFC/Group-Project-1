"""CSCI 270 Group Project 1.

Each teammate's work should be placed under the appropriate part marker. This
file must run from top to bottom for the final submission.
"""


# === PART 1: CLASSIFICATION ===
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix, recall_score, precision_score, f1_score, classification_report

data = load_breast_cancer()

print ("\nClass name (index -> label) ")
for i, name in enumerate(data.target_names):
    print(f"{i} = {name} ")

X = data.data
y = data.target
features = data.feature_names

#Explore the data using a data frame
df = pd.DataFrame(X, columns=data.feature_names )
df['target'] = y
df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})

print("\nClass Counts:")
print(df['diagnosis'].value_counts())

z_score = StandardScaler()
X_scaled = z_score.fit_transform(X)


corre_mat = pd.DataFrame(X_scaled, columns=features).corr()
cluster_grid1 = sns.clustermap(
    corre_mat,
    cmap="coolwarm",
    center = 0,
    figsize=(12, 12),
    linewidths=0.5,
    dendrogram_ratio = (0.15, 0.15)
)
cluster_grid1.figure.suptitle('Multicollinearity Correlation Matrix', fontsize=10, y=1.0)
cluster_grid1.ax_heatmap.set_xticklabels(cluster_grid1.ax_heatmap.get_xmajorticklabels(), fontsize=8, rotation=90)
cluster_grid1.ax_heatmap.set_yticklabels(cluster_grid1.ax_heatmap.get_ymajorticklabels(),fontsize=8, rotation=0)
plt.show()


'''Let's get rid of the features without enough multicolinearity with other features'''
features_to_rid = [
    'texture error',
    'smoothness error',
    'symmetery error',
    'mean fractal dimension',
    'fractal dimension error'
]

relevant_features = [f for f in features if f not in features_to_rid]

feature_indices_pruned = [list (features).index(f) for f in relevant_features]
X_pruned = X_scaled[:, feature_indices_pruned]

print(f"Features pruned: 30 -> {X_pruned.shape[1]} Relevant features retained.")


corr_matrix = pd.DataFrame(X_pruned, columns=relevant_features).corr()
cluster_grid2 = sns.clustermap(
    corr_matrix, 
    cmap='coolwarm',
    center=0,
    figsize=(12, 12),
    linewidths=0.5,
    dendrogram_ratio=(0.15, 0.15)
)

cluster_grid2.figure.suptitle('Multicollinearity Correlation Matrix', fontsize=14, y=1.0)
cluster_grid2.ax_heatmap.set_xticklabels(cluster_grid2.ax_heatmap.get_xmajorticklabels(), fontsize=8, rotation=90)
cluster_grid2.ax_heatmap.set_yticklabels(cluster_grid2.ax_heatmap.get_ymajorticklabels(), fontsize=8, rotation=0)
plt.show()

#-----Get the VIF of the pruned features-------
vif_data = pd.DataFrame()
vif_data["Features"] = relevant_features
vif_data["VIF"] = [variance_inflation_factor(X_pruned, i) for i in range(X_pruned.shape[1]) ]

print("====Variance Inflation Factor Diagnostic on Pruned Data====")
print(vif_data.sort_values(by="VIF", ascending=False).to_string(index=False))

'''The resutls from the diagnostics signifies severe multicolllinearity between features. The feaures
carry overlapping information which cause a linear model to struggle with isolating the individual 
impact of the features. This makes standard regression coeffiecients unreliable for tracking true feature
importance. Hence, we will employ PCA to squish the shared patterns of the features into two super variables,
PC1, and PC2. This allows the model to efficiently understand the data and the patterns within it'''



redundant_features = [
    'mean radius', 'mean perimeter', 'mean area', 
    'worst radius', 'worst perimeter', 'worst area',
    'radius error', 'perimeter error', 'area error', 
    'mean concavity', 'mean concave points', 'mean compactness',
    'worst concavity', 'worst concave points', 'worst compactness',
]

#Extract their column indices from the full feature list
feature_indices = [list(features).index(feature) for feature in redundant_features]
X_redundant = X_scaled[:, feature_indices]

#Fit PCA with two components on the redundant features
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_redundant)

#structure the loadings matrix
loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(2)],
    index=redundant_features
)

print("\n_______Structural Profile Component Matrix_______")
print(loadings_df.round(3))

#Check how much information is preserved by the two principal components
explained_variance = pca.explained_variance_ratio_
print("\n--- PCA Explained Variance on the Redundant Features ---")
print(f"Component 1 (Tumor Size): {explained_variance[0]:.4%} ")
print(f"Component 2 (Border Irregularity Index): {explained_variance[1]:.2%} ")
print(f"Total Information Retained: {np.sum(explained_variance):.2%} ")

#Check the VIF of the 2 newly synthesized features to ensure that they are not collinear
R = np.corrcoef(X_pca, rowvar=False)
vif_new = np.diag(np.linalg.inv(R))

print("\n--- VIF of the 2 Synthesized PCA Features ---")
print(f"VIF Component 1: {vif_new[0]:.4f}")
print(f"VIF Component 2: {vif_new[1]:.4f}")


#Check for OLS coefficients of the 2 PCA features
ols_pca = LinearRegression()
ols_pca.fit(X_pca, y)
print("\n--- OLS Coefficients of the 2 Synthesized PCA Features ---")
print(f"Component 1 (Tumor Size): {ols_pca.coef_[0]:+.4f}")
print(f"Component 2 (Border Irregularity Index): {ols_pca.coef_[1]:+.4f}")

'''The OLS coefficients shows that as the tumor's physical size grows, that is component 1, pr its borders become more
irregular, that is component 2, the model's output decreases toward 0, which is malignant. Which is good because the model
is moving in the right direction of what constitutes as a breast tumor.'''


#=====================
#TRAIN THE DATA
#======================

X_temp, X_test, y_temp, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)

INITIALLY_LABELED = 15

X_train, X_pool, y_train, y_pool = train_test_split(
    X_temp, y_temp, train_size=INITIALLY_LABELED, stratify=y_temp, random_state=42
)


#Active learning loop
N_QUERIES = 30
batch_size = 1

history = {
    'train_size': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1 score': [],
    'report': []
}

model = RandomForestClassifier(n_estimators=2, random_state=42)


for rounds in range(N_QUERIES):
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    model.fit(X_train_balanced, y_train_balanced)

    pred = model.predict(X_test)

    history['train_size'].append(len(X_train))
    history['accuracy'].append(accuracy_score(y_test, pred))
    history['precision'].append(precision_score(y_test, pred, pos_label=0))
    history['recall'].append(recall_score(y_test, pred, pos_label=0))
    history['f1 score'].append(f1_score(y_test, pred, pos_label=0))
    report = classification_report(y_test, pred, target_names=data.target_names)
    history['report'].append(report)

    if len(X_pool) == 0:
        print("\nThe Pool is exhausted")
        break

    pool_prob = model.predict_proba(X_pool)
    uncertainty = 1 - np.max(pool_prob,axis=1)

    query_idx = np.argsort(uncertainty)[-batch_size:]

    X_train = np.vstack([X_train, X_pool[query_idx]])
    y_train = np.concatenate([y_train, y_pool[query_idx]])


    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

    print(f"\nRound {rounds + 1}: train_size = {len(X_train)}",
          f"Malignant precision = {history['precision'][-1]:.3f}, ",
          f"Malignant recall = {history['recall'][-1]:.3f}, ",
          f"Malignant f1 score = {history['f1 score'][-1]:.3f}, ",
          f"Malignant accuracy = {history['accuracy'][-1]:.3f}",
           f"\n----Classification report----\n{report}")

plt.figure(figsize=(12, 12))
plt.plot(history['train_size'],history['accuracy'], marker = '*', label = "Accuracy")
plt.plot(history['train_size'], history['f1 score'], marker="*", label="F1 Score")
plt.plot(history['train_size'], history['precision'], marker="*", label="Precision")
plt.plot(history['train_size'], history['recall'], marker="*", label="Recall")
plt.title("Active Learning Curve for Malignant Prediction")
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.tight_layout()
plt.grid(True, alpha = 0.3)
plt.legend()
plt.show()

confusion_mat = confusion_matrix(y_test, pred)
print("\nConfusion Matrix\n")
print(confusion_mat)


y_test_proba_malignant = model.predict_proba(X_test)[:, 0]

#calculate precision and recall across all thresholds
precisions, recalls, thresholds = precision_recall_curve(
    y_test, y_test_proba_malignant, pos_label=0
)

plt.figure(figsize=(8,6))
plt.plot(recalls, precisions, color ='purple', linewidth=2, label='Random Forest (Malignant)')
plt.title("Precision-Recall curve for Malignant Detection", fontsize=14)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower left')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()



# ============================================================
# CSCI 270 - Project #1
# PART 2: Regression - Housing Prices & The Ceiling Effect
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================

print("\n=== PART 2: HOUSING PRICE REGRESSION ===")

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)


# ============================================================
# STEP 2: IDENTIFY THE CEILING EFFECT
# ============================================================

# Housing values are measured in units of $100,000
# Target values >= 5.0 (specifically 5.00001) represent censored/top-coded data
cap = 5.00000

# Count how many housing values are capped
capped_count = (df["MedHouseVal"] >= cap).sum()

print("\n=== CEILING EFFECT ===")
print("Maximum housing value:", df["MedHouseVal"].max())
print("Number of capped observations:", capped_count)
print("Percentage capped:",
      round(capped_count / len(df) * 100, 2), "%")


# ============================================================
# STEP 3: SPLIT THE DATASET
# ============================================================

# X contains the input features
X = df.drop(columns=["MedHouseVal"])

# y contains the housing prices
y = df["MedHouseVal"]

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\n=== DATA SPLIT ===")
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))


# ============================================================
# STEP 4: BASELINE LINEAR REGRESSION MODEL
# ============================================================

# Create the baseline model
baseline_model = LinearRegression()

# Train using the original data
baseline_model.fit(X_train, y_train)

# Make predictions
baseline_predictions = baseline_model.predict(X_test)

# Calculate performance
baseline_mse = mean_squared_error(
    y_test, baseline_predictions
)

baseline_r2 = r2_score(
    y_test, baseline_predictions
)

print("\n=== BASELINE MODEL RESULTS ===")
print("MSE:", round(baseline_mse, 4))
print("R2 Score:", round(baseline_r2, 4))


# ============================================================
# STEP 5: BASELINE PREDICTED VS ACTUAL SCATTER PLOT
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    baseline_predictions,
    y_test,
    alpha=0.3,
    color="steelblue",
    label="Housing observations"
)

# Highlight the artificial ceiling
plt.axhline(
    y=cap,
    color="red",
    linestyle="--",
    linewidth=2,
    label="$500,001 Price Cap"
)

# Reference identity line (ideal y = x)
min_val = min(baseline_predictions.min(), y_test.min())
max_val = max(baseline_predictions.max(), y_test.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="black",
    linestyle=":",
    linewidth=1.5,
    label="Ideal Prediction (y = x)"
)

plt.xlabel("Predicted Housing Value ($100,000s)")
plt.ylabel("Actual Housing Value ($100,000s)")
plt.title("Baseline Model: Predicted vs Actual Prices")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save the graph for the report
plt.savefig("part2_baseline_scatter.png", dpi=300)
plt.show()


# ============================================================
# STEP 6: CLEAN THE CAPPED DATA
# ============================================================

print("\n=== DATA CLEANING ===")

# Identify uncapped training observations
mask = y_train < cap

# Remove capped observations from training data
X_train_clean = X_train.loc[mask]
y_train_clean = y_train.loc[mask]

print("Original training size:", len(X_train))
print("Cleaned training size:", len(X_train_clean))
print("Removed observations:",
      len(X_train) - len(X_train_clean))

# Keep the original test set for a fair comparison


# ============================================================
# STEP 7: TRAIN THE CLEANED REGRESSION MODEL
# ============================================================

cleaned_model = LinearRegression()

# Train using only uncapped observations
cleaned_model.fit(X_train_clean, y_train_clean)

# Predict using the SAME test set
cleaned_predictions = cleaned_model.predict(X_test)

# Calculate cleaned model performance
cleaned_mse = mean_squared_error(
    y_test, cleaned_predictions
)

cleaned_r2 = r2_score(
    y_test, cleaned_predictions
)

print("\n=== CLEANED MODEL RESULTS ===")
print("MSE:", round(cleaned_mse, 4))
print("R2 Score:", round(cleaned_r2, 4))


# ============================================================
# STEP 8: EVALUATE PERFORMANCE ON UNCAPPED TEST HOMES
# ============================================================

# Evaluate only test homes below the price cap
uncapped_test = y_test < cap

baseline_uncapped_mse = mean_squared_error(
    y_test.loc[uncapped_test],
    baseline_predictions[uncapped_test]
)

cleaned_uncapped_mse = mean_squared_error(
    y_test.loc[uncapped_test],
    cleaned_predictions[uncapped_test]
)

baseline_uncapped_r2 = r2_score(
    y_test.loc[uncapped_test],
    baseline_predictions[uncapped_test]
)

cleaned_uncapped_r2 = r2_score(
    y_test.loc[uncapped_test],
    cleaned_predictions[uncapped_test]
)

print("\n=== UNCAPPED TEST HOMES ===")
print("Baseline MSE:", round(baseline_uncapped_mse, 4))
print("Cleaned MSE:", round(cleaned_uncapped_mse, 4))
print("Baseline R2:", round(baseline_uncapped_r2, 4))
print("Cleaned R2:", round(cleaned_uncapped_r2, 4))


# ============================================================
# STEP 9: COMPARE BOTH MODELS
# ============================================================

comparison = pd.DataFrame({
    "Model": ["Baseline", "Cleaned"],
    "MSE": [baseline_mse, cleaned_mse],
    "R2": [baseline_r2, cleaned_r2]
})

print("\n=== MODEL COMPARISON (FULL TEST SET) ===")
print(comparison.to_string(index=False))

# Full comparison table for defense report
comparison_full = pd.DataFrame({
    "Evaluation Set": ["Full Test Set", "Full Test Set", "Uncapped Test Homes", "Uncapped Test Homes"],
    "Model": ["Baseline", "Cleaned", "Baseline", "Cleaned"],
    "MSE": [baseline_mse, cleaned_mse, baseline_uncapped_mse, cleaned_uncapped_mse],
    "R2": [baseline_r2, cleaned_r2, baseline_uncapped_r2, cleaned_uncapped_r2]
})

print("\n=== COMPREHENSIVE PERFORMANCE COMPARISON ===")
print(comparison_full.to_string(index=False))

comparison_full.to_csv("part2_model_comparison.csv", index=False)


# ============================================================
# STEP 10: MEASURE PREDICTION BIAS
# ============================================================

# Positive error means the model overpredicts
baseline_errors = (
    baseline_predictions[uncapped_test]
    - y_test.loc[uncapped_test].to_numpy()
)

cleaned_errors = (
    cleaned_predictions[uncapped_test]
    - y_test.loc[uncapped_test].to_numpy()
)

print("\n=== PREDICTION BIAS ON UNCAPPED HOMES ===")
print("Baseline mean error:",
      round(baseline_errors.mean(), 4))
print("Cleaned mean error:",
      round(cleaned_errors.mean(), 4))


# ============================================================
# STEP 11: CLEANED MODEL SCATTER PLOT
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    cleaned_predictions,
    y_test,
    alpha=0.3,
    color="green",
    label="Housing observations"
)

plt.axhline(
    y=cap,
    color="red",
    linestyle="--",
    linewidth=2,
    label="$500,001 Price Cap"
)

# Reference identity line (ideal y = x)
min_clean_val = min(cleaned_predictions.min(), y_test.min())
max_clean_val = max(cleaned_predictions.max(), y_test.max())
plt.plot(
    [min_clean_val, max_clean_val],
    [min_clean_val, max_clean_val],
    color="black",
    linestyle=":",
    linewidth=1.5,
    label="Ideal Prediction (y = x)"
)

plt.xlabel("Predicted Housing Value ($100,000s)")
plt.ylabel("Actual Housing Value ($100,000s)")
plt.title("Cleaned Model: Predicted vs Actual Prices")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("part2_cleaned_scatter.png", dpi=300)
plt.show()


# ============================================================
# STEP 12: PERFORMANCE COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["Model"],
    comparison["MSE"],
    color=["steelblue", "green"]
)

plt.ylabel("Mean Squared Error")
plt.title("Baseline vs Cleaned Model: MSE")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("part2_mse_comparison.png", dpi=300)
plt.show()


# ============================================================
# STEP 13: FINAL SUMMARY
# ============================================================

print("\n=== FINAL SUMMARY ===")

if cleaned_mse < baseline_mse:
    print("The cleaned model has a lower overall MSE.")
else:
    print("The baseline model has a lower or equal overall MSE.")

if cleaned_uncapped_mse < baseline_uncapped_mse:
    print("Cleaning improved MSE on uncapped test homes.")
else:
    print("Cleaning did not improve MSE on uncapped test homes.")

print("\nPart 2 completed successfully!")


# === PART 3: CLUSTERING ===
# Teammate code goes here.


# === PART 4: TIME SERIES - AIR QUALITY AND THE AI AUDIT ===

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATION = "Aotizhongxin"
POLLUTANT = "PM2.5"



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
