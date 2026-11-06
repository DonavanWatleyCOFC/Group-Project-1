
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
