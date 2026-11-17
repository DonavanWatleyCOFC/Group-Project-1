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