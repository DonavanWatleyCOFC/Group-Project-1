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

"""
Part Three: Customer Segmentation using K-Means Clustering
Dataset: Mall Customer Segmentation Data (Mall_Customers.csv)

Engineering Tasks:
1. Load CSV using Pandas and drop uninformative identifier columns (CustomerID).
2. Run K-Means clustering using two distinct feature combinations:
   - Combo 1: Annual Income vs. Spending Score (Canonical baseline)
   - Combo 2: Age vs. Spending Score (Demographic profile)
   - (Bonus Extension: 3D feature set Age + Income + Spending Score)
3. Programmatically calculate Inertia (Elbow Method) and Silhouette Scores across K in [2, 10].
4. Analyze and visualize the mathematical justification for K=5 vs K=6.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath: str = "Mall_Customers.csv") -> pd.DataFrame:
    """Load customer dataset and drop uninformative identifier columns."""
    df = pd.read_csv(filepath)
    print("=" * 70)
    print("1. DATA LOADING & PREPROCESSING")
    print("=" * 70)
    print(f"Loaded dataset from '{filepath}' with shape {df.shape}")
    print(f"Original Columns: {list(df.columns)}")

    # Drop uninformative identifier column
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
        print("Dropped uninformative column: 'CustomerID'")

    print("\nCleaned Data Summary:")
    print(df.info())
    print("\nSample Rows:")
    print(df.head())
    print("-" * 70)
    return df


def evaluate_kmeans_range(
    data: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """Programmatically compute Inertia (WCSS) and Silhouette Score for K in k_range."""
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        cluster_labels = kmeans.fit_predict(data)
        inertia = kmeans.inertia_
        sil_score = silhouette_score(data, cluster_labels)

        results.append(
            {
                "K": k,
                "Inertia (WCSS)": round(inertia, 2),
                "Silhouette Score": round(sil_score, 4),
            }
        )

    return pd.DataFrame(results)


def plot_elbow_and_silhouette(
    results_combo1: pd.DataFrame,
    results_combo2: pd.DataFrame,
    label1: str,
    label2: str,
    save_path: str = "clustering_evaluation.png",
):
    """Plot Elbow Method (Inertia) and Silhouette Scores side by side for both feature combinations."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Plot 1: Elbow Method (Inertia / WCSS) ---
    axes[0].plot(
        results_combo1["K"],
        results_combo1["Inertia (WCSS)"],
        "o-",
        color="#1f77b4",
        linewidth=2,
        label=label1,
    )
    axes[0].plot(
        results_combo2["K"],
        results_combo2["Inertia (WCSS)"],
        "s--",
        color="#ff7f0e",
        linewidth=2,
        label=label2,
    )
    axes[0].set_title(
        "Elbow Method: Inertia vs. K", fontsize=13, fontweight="bold"
    )
    axes[0].set_xlabel("Number of Clusters (K)", fontsize=11)
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    axes[0].set_xticks(results_combo1["K"])
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Annotate K=5 and K=6 ambiguity
    axes[0].axvline(
        x=5, color="gray", linestyle=":", alpha=0.7, label="Candidate K=5"
    )
    axes[0].axvline(
        x=6, color="purple", linestyle=":", alpha=0.7, label="Candidate K=6"
    )

    # --- Plot 2: Silhouette Score ---
    axes[1].plot(
        results_combo1["K"],
        results_combo1["Silhouette Score"],
        "o-",
        color="#1f77b4",
        linewidth=2,
        label=label1,
    )
    axes[1].plot(
        results_combo2["K"],
        results_combo2["Silhouette Score"],
        "s--",
        color="#ff7f0e",
        linewidth=2,
        label=label2,
    )
    axes[1].set_title(
        "Silhouette Score vs. K", fontsize=13, fontweight="bold"
    )
    axes[1].set_xlabel("Number of Clusters (K)", fontsize=11)
    axes[1].set_ylabel("Silhouette Score", fontsize=11)
    axes[1].set_xticks(results_combo1["K"])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved evaluation plots to: {save_path}")
    plt.close()


def plot_cluster_comparisons(
    df: pd.DataFrame,
    features_combo1: list[str],
    features_combo2: list[str],
    save_path: str = "cluster_segmentations.png",
):
    """Plot customer segmentation scatter plots comparing K=5 and K=6 for both combinations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    combos = [
        (features_combo1, "Annual Income vs Spending Score"),
        (features_combo2, "Age vs Spending Score"),
    ]

    for row_idx, (features, title_prefix) in enumerate(combos):
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for col_idx, k in enumerate([5, 6]):
            ax = axes[row_idx, col_idx]
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)

            # Scatter plot using original scales for intuitive interpretation
            scatter = ax.scatter(
                X[:, 0],
                X[:, 1],
                c=labels,
                cmap="tab10",
                s=60,
                alpha=0.85,
                edgecolors="none",
            )

            # Inverse-transform cluster centers for plotting on original scale
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c="black",
                marker="X",
                s=180,
                label="Centroids",
            )

            ax.set_title(
                f"{title_prefix} (K={k}, Silhouette={sil:.3f})",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel(features[0], fontsize=10)
            ax.set_ylabel(features[1], fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved cluster comparison plots to: {save_path}")
    plt.close()


def main():
    # 1. Load Data
    data_path = Path("Mall_Customers.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path.resolve()}'."
        )

    df = load_and_preprocess_data(str(data_path))

    # 2. Define Distinct Feature Combinations
    # Combination 1: Canonical Income vs Spending Score
    combo1_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    # Combination 2: Demographic Age vs Spending Score
    combo2_cols = ["Age", "Spending Score (1-100)"]

    print("\n" + "=" * 70)
    print("2. FEATURE COMBINATIONS FOR CLUSTERING")
    print("=" * 70)
    print(f"Feature Combination 1 (Canonical):   {combo1_cols}")
    print(f"Feature Combination 2 (Demographic): {combo2_cols}")

    # Standardize features for sound distance-based clustering
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(df[combo1_cols])
    X2_scaled = scaler.fit_transform(df[combo2_cols])

    # 3. Programmatic Inertia & Silhouette Evaluation Across K Range
    k_range = range(2, 11)
    results1 = evaluate_kmeans_range(X1_scaled, k_range=k_range)
    results2 = evaluate_kmeans_range(X2_scaled, k_range=k_range)

    print("\n" + "=" * 70)
    print("3. PROGRAMMATIC EVALUATION: INERTIA & SILHOUETTE SCORES")
    print("=" * 70)

    print("\n--- Feature Combination 1: Annual Income vs. Spending Score ---")
    print(results1.to_string(index=False))

    best_k1 = results1.loc[results1["Silhouette Score"].idxmax()]["K"]
    best_sil1 = results1["Silhouette Score"].max()
    print(
        f">> Optimal K by Silhouette Score for Combo 1: K={int(best_k1)} (Score: {best_sil1:.4f})"
    )

    print("\n--- Feature Combination 2: Age vs. Spending Score ---")
    print(results2.to_string(index=False))

    best_k2 = results2.loc[results2["Silhouette Score"].idxmax()]["K"]
    best_sil2 = results2["Silhouette Score"].max()
    print(
        f">> Optimal K by Silhouette Score for Combo 2: K={int(best_k2)} (Score: {best_sil2:.4f})"
    )

    # 4. Generate Visualizations
    print("\n" + "=" * 70)
    print("4. GENERATING VISUALIZATIONS")
    print("=" * 70)
    plot_elbow_and_silhouette(
        results_combo1=results1,
        results_combo2=results2,
        label1="Combo 1: Income vs Spending",
        label2="Combo 2: Age vs Spending",
        save_path="clustering_evaluation.png",
    )

    plot_cluster_comparisons(
        df=df,
        features_combo1=combo1_cols,
        features_combo2=combo2_cols,
        save_path="cluster_segmentations.png",
    )

    # 5. Engineering Justification: The K=5 vs K=6 Ambiguity
    print("\n" + "=" * 70)
    print("5. ENGINEERING ANALYSIS: MATHEMATICAL JUSTIFICATION FOR K=5 vs. K=6")
    print("=" * 70)
    analysis = """
    Why the Elbow Method & Silhouette Scores Justify K=5 vs. K=6:

    1. Case for K=5 (Supported by Annual Income vs. Spending Score):
       - When clustering on [Annual Income, Spending Score], K=5 yields the global 
         maximum Silhouette Score (~0.555).
       - Business intuition aligns with 5 clean quadrants + center:
         (1) Low Income / Low Spending (Sensible / Budget)
         (2) Low Income / High Spending (Careless)
         (3) High Income / Low Spending (Frugal / Savers)
         (4) High Income / High Spending (Affluent / Target)
         (5) Moderate Income / Moderate Spending (Balanced Middle)
       - At K=5, the elbow curve shows a sharp reduction in rate of inertia loss.

    2. Case for K=6 (Supported by Age vs. Spending Score or 3D feature sets):
       - In [Age, Spending Score] under standard scaling, K=6 achieves the peak 
         silhouette score (~0.457), outperforming K=5 (~0.448).
       - When incorporating Age, the central 'moderate spender' cluster naturally 
         bifurcates across age demographics (e.g., Young Moderate vs. Senior Moderate), 
         which is critical for generational targeting and product personalization.
       - Consequently, while K=5 is optimal for pure financial segmentation, K=6 is 
         mathematically justified when age dynamics inform the customer profile.
    """
    print(analysis.strip())
    print("=" * 70)




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
