"""CSCI 270 Group Project 1.

Each teammate's work should be placed under the appropriate part marker. This
file must run from top to bottom for the final submission.
"""


# === PART 1: CLASSIFICATION ===
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix, recall_score, precision_score, f1_score

data = load_breast_cancer()

print ("\nClass name (index -> label) ")
for i, name in enumerate(data.target_names):
    print(f"{i} = {name} ")

X = data.data
y = data.target

#Explore the data using a data frame
df = pd.DataFrame(X, columns=data.feature_names )
df['target'] = y
df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})

print("\nClass Counts:")
print(df['diagnosis'].value_counts())


#Transpowse .describe() to get summary of dataset and format features to be rows to look at min, max, and std
stats = df.drop(columns=['target', 'diagnosis']).describe().T[['min', 'max', 'std']]
print("\nFeature Statistics: ")
print(stats.sort_values(by='std', ascending=False))


#Normalize the dataset
zscore = StandardScaler()
X_scaled = zscore.fit_transform(X)

#PCA for visualization
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled)

#Visualize the PCA results using seaborn scatter plot
plt.figure(figsize=(8,6)) #To set up the canvas size

#seaborn draws the data
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df['diagnosis'],
    alpha=0.7
)
plt.title("PCA of Breast Cancer Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

print(f"\nTotal Variance Captured by PCA: {np.sum(pca.explained_variance_ratio_):.2f}")

#Try DBSCAN


#=====================
#TRAIN THE DATA
#======================

#1. Fix held-out set. This is only for measuring progress and will nevery be queried during the active learning
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
#Note: Stratify means to preserve the exact same proportion(ratio) of classes when splitting you dataset into
# training and testing sets. This allows for fair evalution and prevents rare class starvation 

#1.1. From the remainig data: small initial labeled set + large unlabeled pool
INITIAL_LABELED = 30 # starting training size


smote = SMOTE(random_state=42)

X_temp_bal, y_temp_bal = smote.fit_resample(X_temp, y_temp)

X_train, X_pool, y_train, y_pool = train_test_split(
    X_temp_bal, y_temp_bal, train_size=INITIAL_LABELED, stratify=y_temp_bal, random_state=42
)


print(f"\nInitial train size: {len(X_train)}")#the size of the training data
print(f"Pool size: {len(X_pool)}")# the size of the data we will be querying from
print(f"Held-out test size: {len(X_test)}")# the fixed test data that is used to measure the model's performance

#=====================================
#2. Active Learning Loop
#=====================================
N_QUERIES = 15 # The number of queries to run
BATCH_SIZE = 2 # The the number of pool points to label per round

history = {
    'train_size':[],
    'accuracy':[],
    'precision':[],
    'recall':[],
    'f1':[]

}

#n_estimators: The number of individual decision trees built in the forest
model = RandomForestClassifier(n_estimators=100, random_state=42)


for num_rounds in range(N_QUERIES):
    model.fit(X_train, y_train) #train on current labeled set

    #evaluate on the FIXED test set
    pred = model.predict(X_test)
    history['train_size'].append(len(X_train))
    history['accuracy'].append(accuracy_score(y_test, pred))
    history['precision'].append(precision_score(y_test, pred, pos_label=0))
    history['recall'].append(recall_score(y_test,pred, pos_label=0))
    history['f1'].append(f1_score(y_test, pred, pos_label=0))

    if len(X_pool) == 0:
        print("\nThe pool is exhausted")
        break
    #Query strategy, uncertainty sampling
    #For each pool point, get predicted class probabilities
    #'Uncertainty' = how close the top probability is to 0.5 (max entropy for binary)
    pool_proba = model.predict_proba(X_pool)
    uncertainty = 1 - np.max(pool_proba, axis=1) #closer to 0.5 max-prob -> higher uncertainty 

    #Pick the BATCH_SIZE most uncertain points
    query_idx = np.argsort(uncertainty)[-BATCH_SIZE:]

    #Label the data (since this is a benchmark dataset already have the true labels)
    X_train = np.vstack([X_train, X_pool[query_idx]])
    y_train = np.concatenate([y_train, y_pool[query_idx]])

    #Remove queried points from the pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)


    print(f"\nRound {num_rounds + 1}: train_size={len(X_train)}, "
          f"Malignant recall = {history['recall'][-1]:.3f}, "
          f"Malignant precision = {history['precision'][-1]:.3f}, "
          f"Accuracy = {history['accuracy'][-1]:.3f}")

#============================
#3. Plot learning curve
#============================
plt.figure(figsize=(9,6))
plt.plot(history['train_size'], history['accuracy'], marker='o',label='Accuracy')
plt.plot(history['train_size'], history['precision'], marker='o', label='Precision, malignant')
plt.plot(history['train_size'], history['recall'], marker='o', label='Recall, malignant')
plt.plot(history['train_size'], history['f1'], marker='o', label='F1 score, malignant')
plt.xlabel("Training set size")
plt.ylabel("Score")
plt.title("Active learning Progress")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

from sklearn.metrics import precision_recall_curve
#get predicted probabilities for the test set
#column 0 is probability of malignant, column 1 is benign
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

conusion_mat = confusion_matrix(y_test, pred)
print("\nConfusion Matrix\n")
print(conusion_mat)



#Try DBScan 
#. Import the necessary libraries
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

#2. To determing the optimal epsilon value for DBSCAN using k-distance graph
k = 5 #number of nearest neighbors to consider
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)

#Get the distances to nearest neighbors for all points on the dataset
distances, _ = neighbors.kneighbors(X_scaled)

#3. Sort the k-th neighbor distances from smallest to largest
k_distances = np.sort(distances[:, -1]) 

#4. Plot the sorted k-distances 
plt.figure(figsize=(8,6))
plt.plot(k_distances)
plt.title(f"K-distance Graph (k={k}) to Find optimal epsilon for DBSCAN")
plt.xlabel("Data points sorted by distance to their 5th nearest neighbor")
plt.ylabel(f"{k}-NN Distance (Epsilon candidate)")
#True turns on the background grid lines, alpha controls the transparency of the grid lines  on scale from 0.0 to 1.0
plt.grid(True, alpha = 0.3)
plt.show()

print("Point with distance <= 3.5:", np.sum(k_distances <= 3.5))
print("Points with distance > 3.5 (Noise):", np.sum(k_distances > 3.5))

dbscan = DBSCAN(eps=3.5, min_samples=k)
labels = dbscan.fit_predict(X_scaled)

#Check how many clusters and how many noise points (-1) were found 
unique_labels, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count} points")

# Find out what Diagnosis classes the 75 noise points belong to
print("\nUnmasking the 75 Noise points: ")
print(pd.crosstab(df['diagnosis'], labels, rownames=['True Diagnosis'], colnames=['DBSCAN Cluster']))

# Compare the average cell size/shape of Cluster 0 vs Noise (-1)
comparison = df.drop(columns=['diagnosis', 'target']).groupby(labels).mean().T
comparison.columns = ['Noise Outliers (-1)', 'Main Cluster (0)']
comparison['Ratio (Noise / Cluster 0)'] = comparison['Noise Outliers (-1)'] / comparison['Main Cluster (0)']

print("\n--- Key Features Driving the Noise Outliers ---")
print(comparison[['Main Cluster (0)', 'Noise Outliers (-1)', 
                  'Ratio (Noise / Cluster 0)']].sort_values(by='Ratio (Noise / Cluster 0)', ascending=False).head(5))



#Evaluate DBSCAN 
from sklearn.metrics import adjusted_rand_score

#1. Anomaly Precision: Out of points flagged as noise, how many were actual cancer?
noise_mask = (labels == -1) 
malignant_mask = (df['diagnosis'] == 'malignant')

anomaly_precision = np.sum(noise_mask & malignant_mask) / np.sum(noise_mask)
print(f"Anomaly Precision: {anomaly_precision:.1%}")

#2. Adjusted rand index (measures how well clusters match true diagnosis, 0 to 1)
ari = adjusted_rand_score(df['target'], labels)
print(f"Adjusted Rand Index (ARI): {ari:.3f}")


# Create custom color and marker map for DBSCAN
dbscan_names = {0: 'Main Cluster (0)', -1: 'Noise Outlier (-1)'}
dbscan_series = pd.Series(labels).map(dbscan_names)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left: Ground Truth
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1], 
    hue=df['diagnosis'], palette={'benign': 'tab:orange', 'malignant': 'tab:blue'},
    alpha=0.7, ax=ax1
)
ax1.set_title("Ground Truth Diagnoses", fontsize=13)
ax1.set_xlabel("Principal Component 1")
ax1.set_ylabel("Principal Component 2")
ax1.grid(True, alpha=0.3)

# Right: DBSCAN Clusters + Noise
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1], 
    hue=dbscan_series, style=dbscan_series,
    palette={'Main Cluster (0)': 'steelblue', 'Noise Outlier (-1)': 'crimson'},
    markers={'Main Cluster (0)': 'o', 'Noise Outlier (-1)': 'X'},
    s=50, alpha=0.8, ax=ax2
)
ax2.set_title("DBSCAN Clustering (eps=3.5, min_samples=5)", fontsize=13)
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


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
