from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix, recall_score, precision_score, f1_score, classification_report

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

print("\nFirst 20 binary labels:")
print(y[:20])


#VISUALIZE THE DATASET
#1. Class balance
plt.figure(figsize=(8,6))
sns.countplot(x='diagnosis', data=df, order=['malignant', 'benign'])
plt.title("Class Distribution")
plt.tight_layout()
plt.show()

#2. Distribution of key features
key_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
    'mean concavity'
]
#this creates a window with a 2 by 3 matrix arry of individual plots of
fig, axes = plt.subplots(2, 3, figsize=(15,8))
#this loop flattens the 2D grid of subplots, pairs each plot box (ax) with a faature name(Feat) so that they are processed together
for ax, feat in zip(axes.flat, key_features):
    #this draws a kernel density estimation plot (a smoothed-out histogram showing data distribution)
    sns.kdeplot(data=df, x=feat, hue='diagnosis', fill=True, ax=ax)
    #this sets teh title on top of each subplot to the name of that feature
    ax.set_title(feat)
plt.tight_layout() #fixes spacing so axis labels and titles don't collide
plt.show()# displays the figure

#3. Raw Data Points Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='mean radius', y='mean texture', hue='diagnosis', alpha=0.7)
plt.title("Raw Data Points: Mean Radius vs. Mean Texture")
plt.tight_layout()
plt.show()


#FEATURE TO CLASS CORRELATION
#correlate each feature to the numerical target
correlations = df.drop(columns='diagnosis').corr()['target'].drop('target').sort_values()

# print("\nFeature correlation with target (0 = malignant, 1 = benign)")
# print(correlations)

#visualize correlation
plt.figure(figsize=(8, 10))
correlations.plot(kind='barh',color=correlations.apply(lambda v: 'crimson' if v < 0 else 'steelblue'))
plt.title("Feature Correlation with Diagnosis \n(negative = higher in malignant tumors")
plt.xlabel("Correlation coefficient")
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

#full correlation heat map
plt.figure(figsize=(16,14))
sns.heatmap(df.drop(columns='diagnosis').corr(),cmap='coolwarm',center=0, annot=False)
plt.title("Feature Correlation Heat Map")
plt.tight_layout()
plt.show()


#=====================
#TRAIN THE DATA (Balanced 100 points, averaged over 10 iterations)
#======================

N_ITERATIONS = 10
N_QUERIES = 15
BATCH_SIZE = 2
INITIAL_LABELED = 20

all_train_sizes = []
all_accuracies = np.zeros((N_ITERATIONS, N_QUERIES))
all_precisions = np.zeros((N_ITERATIONS, N_QUERIES))
all_recalls = np.zeros((N_ITERATIONS, N_QUERIES))
all_f1s = np.zeros((N_ITERATIONS, N_QUERIES))

all_y_test = []
all_y_test_proba_malignant = []

from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

for iteration in range(N_ITERATIONS):
    # Create a balanced dataset of 100 points (50 malignant, 50 benign)
    malignant_idx = np.where(y == 0)[0]
    benign_idx = np.where(y == 1)[0]
    
    mal_chosen = np.random.choice(malignant_idx, 50, replace=False)
    ben_chosen = np.random.choice(benign_idx, 50, replace=False)
    
    balanced_idx = np.concatenate([mal_chosen, ben_chosen])
    np.random.shuffle(balanced_idx) # shuffle to mix classes
    
    X_bal = X[balanced_idx]
    y_bal = y[balanced_idx]

    # 1. Fix held-out set. 
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=None
    )
    
    # 1.1. From the remainig data: small initial labeled set + large unlabeled pool
    X_train, X_pool, y_train, y_pool = train_test_split(
        X_temp, y_temp, train_size=INITIAL_LABELED, stratify=y_temp, random_state=None
    )
    
    if iteration == 0:
        print(f"\nExample Iteration Info:")
        print(f"Initial train size: {len(X_train)}")
        print(f"Pool size: {len(X_pool)}")
        print(f"Held-out test size: {len(X_test)}")

    model = RandomForestClassifier(n_estimators=100, random_state=None)
    
    iter_train_sizes = []
    
    for num_rounds in range(N_QUERIES):
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        iter_train_sizes.append(len(X_train))
        
        all_accuracies[iteration, num_rounds] = accuracy_score(y_test, pred)
        all_precisions[iteration, num_rounds] = precision_score(y_test, pred, zero_division=0)
        all_recalls[iteration, num_rounds] = recall_score(y_test, pred, zero_division=0)
        all_f1s[iteration, num_rounds] = f1_score(y_test, pred, zero_division=0)
        
        if len(X_pool) == 0:
            break
            
        pool_proba = model.predict_proba(X_pool)
        uncertainty = 1 - np.max(pool_proba, axis=1)
        query_idx = np.argsort(uncertainty)[-BATCH_SIZE:]
        
        X_train = np.vstack([X_train, X_pool[query_idx]])
        y_train = np.concatenate([y_train, y_pool[query_idx]])
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

    if iteration == 0:
        all_train_sizes = iter_train_sizes
        
    all_y_test.extend(y_test)
    all_y_test_proba_malignant.extend(model.predict_proba(X_test)[:, 0])

# Calculate averages
avg_accuracies = np.mean(all_accuracies, axis=0)
avg_precisions = np.mean(all_precisions, axis=0)
avg_recalls = np.mean(all_recalls, axis=0)
avg_f1s = np.mean(all_f1s, axis=0)

print(f"\nAveraged Final Stats after {N_ITERATIONS} iterations:")
print(f"Accuracy: {avg_accuracies[-1]:.3f}, Precision: {avg_precisions[-1]:.3f}, Recall: {avg_recalls[-1]:.3f}")

#============================
#3. Plot learning curve
#============================
plt.figure(figsize=(9,6))
plt.plot(all_train_sizes, avg_accuracies, marker='o',label='Avg Accuracy')
plt.plot(all_train_sizes, avg_precisions, marker='o', label='Avg Precision, malignant')
plt.plot(all_train_sizes, avg_recalls, marker='o', label='Avg Recall, malignant')
plt.plot(all_train_sizes, avg_f1s, marker='o', label='Avg F1 score, malignant')
plt.xlabel("Training set size")
plt.ylabel("Score")
plt.title(f"Active Learning Progress (Avg {N_ITERATIONS} Iterations, 100 Balanced Points)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#calculate precision and recall across all thresholds
precisions, recalls, thresholds = precision_recall_curve(
    all_y_test, all_y_test_proba_malignant, pos_label=0
)

plt.figure(figsize=(8,6))
plt.plot(recalls, precisions, color ='purple', linewidth=2, label='Random Forest (Malignant)')

plt.title("Averaged Precision-Recall curve for Malignant Detection", fontsize=14)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower left')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

conusion_mat = confusion_matrix(y_test, pred)
print("\nConfusion Matrix (From Last Iteration)\n")
print(conusion_mat)

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(
    model, 
    X_test, 
    y_test, 
    display_labels=['Malignant (0)', 'Benign (1)'],
    cmap='Blues'
)
plt.title("Confusion Matrix on Fixed Test Set (Last Iteration)")
plt.grid(False)
plt.tight_layout()
plt.show()



#Quantify the exact drop in overall accuracy in the dataset to prioritize recall
    








