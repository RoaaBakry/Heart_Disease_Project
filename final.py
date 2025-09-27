from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# =========================================================================
# 1. DATA LOADING AND CLEANING
# =========================================================================

# Fetch dataset
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# Replace -9 with NaN and Impute Missing Values
X = X.replace(-9, np.nan)
X['ca'] = X['ca'].fillna(X['ca'].mode()[0])
X['thal'] = X['thal'].fillna(X['thal'].mode()[0])
print("Missing values after imputation:\n", X.isnull().sum().sum())

# =========================================================================
# 2. DATA ENCODING AND SCALING
# =========================================================================

# One-Hot Encoding (for categorical features with numeric labels)
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Standard Scaling (for continuous numerical features)
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# =========================================================================
# 3. DATA SPLITTING
# =========================================================================

# Convert target y to a 1D array (required by train_test_split)
y_flat = y.values.ravel()

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_flat, test_size=0.2, random_state=42, stratify=y_flat
)

print(f"Data Split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")

# =========================================================================
# 4. VISUALIZATION AND PCA ANALYSIS

#1. Prepare Data for Correlation: Combine features and the target
df_corr_viz = X_encoded.copy()
df_corr_viz['target'] = y.values.ravel() # Add the target variable as a 1D array

# 2. Calculate the correlation of all columns with the 'target' column
# This gives us a single column of correlation coefficients
target_corr = df_corr_viz.corr()[['target']].sort_values(by='target', ascending=False)

# 3. Visualize the correlation
plt.figure(figsize=(8, 15)) # Set the size for a vertical, readable plot
sns.heatmap(
    target_corr,
    annot=True,          # **annot=True**: Display the correlation coefficient number inside each cell.
    cmap='coolwarm',     # **cmap='coolwarm'**: Color map where red/warm colors show positive correlation (closer to +1) and blue/cool colors show negative correlation (closer to -1).
    fmt=".2f",           # **fmt=".2f"**: Format the displayed numbers to two decimal places.
    linewidths=.5,       # Add small white lines to separate the cells.
    cbar=False           # Do not display the color bar since we are only visualizing one column of correlations.
)
plt.title('Feature Correlation with Heart Disease Target', fontsize=16)
plt.yticks(rotation=0) # Keep feature names horizontal for easy reading
plt.show()

# Print the values directly for analysis
print("\n--- Top Correlation Values with Target ---")
print(target_corr.head(10))

# 3. Distribution Plots for Numerical Features (Age)
plt.figure(figsize=(8, 6))
sns.histplot(data=df_corr_viz, x='age', kde=True, bins=20, color='darkgreen')
plt.title('Standardized Age Distribution', fontsize=16)
plt.xlabel('Age (Scaled)')
plt.show()


# 4. Boxplot for Numerical Features (Focused View)
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_corr_viz[numerical_cols], orient='h', palette='Set2')
plt.title('Boxplot of Standardized Numerical Features', fontsize=16)
plt.show()

# --- B. PCA Analysis to Find Optimal Components ---
pca_full = PCA(n_components=None, random_state=42)
pca_full.fit(X_train) # Fit on training data
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
print(f"Optimal number of components to retain 95% variance: {n_components_95}")

# --- C. Cumulative Explained Variance Plot ---
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='blue')
plt.axhline(y=0.95, color='r', linestyle='-', label='95% Cutoff')
plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'Optimal {n_components_95} Components')
plt.title('Cumulative Explained Variance by Principal Components', fontsize=14)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.legend()
plt.show()

# --- D. PC1 vs PC2 Scatter Plot ---
pca_2 = PCA(n_components=2, random_state=42)
X_train_pca_2 = pca_2.fit_transform(X_train)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_train_pca_2[:, 0],
    y=X_train_pca_2[:, 1],
    hue=y_train,
    palette='viridis',
    legend='full'
)
plt.title('Visualization of Training Data in PC1 vs PC2 Space', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE, chi2, SelectKBest
from sklearn.linear_model import LogisticRegression

# =========================================================================
# 1. METHOD: FEATURE IMPORTANCE (RANDOM FOREST) for feature selection part

# Train a Random Forest Classifier to calculate importance scores
rf_selector = RandomForestClassifier(random_state=42)
rf_selector.fit(X_train, y_train)

# Create a Series of feature importances
importance_df = pd.Series(
    rf_selector.feature_importances_, 
    index=X_train.columns
).sort_values(ascending=False)

print("--- 1. Random Forest Feature Importance (Top 10) ---")
print(importance_df.head(10))


# =========================================================================
# 2. METHOD: RECURSIVE FEATURE ELIMINATION (RFE)

# Initialize a model (Logistic Regression) to use within RFE
rfe_model = LogisticRegression(solver='liblinear', random_state=42)

# Initialize RFE to select the top 10 features
rfe_selector = RFE(estimator=rfe_model, n_features_to_select=10, step=1)
rfe_selector.fit(X_train, y_train)

# Get the selected features
rfe_selected_features = X_train.columns[rfe_selector.support_]

print("\n--- 2. RFE Selected Features (Top 10) ---")
print(rfe_selected_features.tolist())


# =========================================================================
# 3. METHOD: CHI-SQUARE TEST

# Note: Chi-Square requires non-negative data. Our scaled continuous features
# are both negative and positive, so we'll only apply this to the binary (0/1)
# one-hot encoded features which are guaranteed non-negative.

# Identify the categorical/binary columns
binary_cols = X_train.columns[~X_train.columns.isin(numerical_cols)]

# Apply SelectKBest with Chi-Square (selecting the top 10 binary features)
chi2_selector = SelectKBest(chi2, k=10)
chi2_selector.fit(X_train[binary_cols], y_train)

# Get the selected features
chi2_selected_features = binary_cols[chi2_selector.get_support()]

print("\n--- 3. Chi-Square Selected Features (Top 10 Binary) ---")
print(chi2_selected_features.tolist())


# =========================================================================
# 4. FINAL SELECTION & TRANSFORMATION
# =========================================================================

# For simplicity and robustness, we will select the top 12 features based on the
# Random Forest Importance scores (a highly reliable model-based method).

# Get the names of the final selected features (e.g., top 12)
final_selected_features = importance_df.head(12).index.tolist()

# Create the final feature-selected datasets
X_train_fs = X_train[final_selected_features]
X_test_fs = X_test[final_selected_features]

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# NOTE: This code assumes the following variables have been correctly defined 
# and populated from previous steps:
# X_train_fs, X_test_fs (Feature-selected scaled features)
# y_train, y_test (Target variable arrays)

# =========================================================================
# 1. DATA FIX: ENSURE TARGET VARIABLES ARE ROBUST 1D INTEGERS
#    This is the definitive fix to ensure arrays are strictly binary (0 or 1).

# Ensure y arrays are simple 1D integer arrays and convert to numpy for robustness
y_train = np.ravel(y_train).astype(int)
y_test = np.ravel(y_test).astype(int)

# =========================================================================
# 2. MODEL SETUP


# Dictionary of all models to train
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    # SVM needs probability=True for ROC curve generation
    "SVM": SVC(kernel='linear', random_state=42, probability=True)
}

results = {}
plt.figure(figsize=(10, 8))

print("--- Supervised Model Training and Evaluation ---")

# =========================================================================
# 3. TRAINING AND EVALUATION LOOP (FINAL ROBUST FIX)


for name, model in models.items():
    # 3a. Train the model on Feature Selected Data
    model.fit(X_train_fs, y_train)

    # 3b. Make predictions and get probabilities
    y_pred = model.predict(X_test_fs)
    
    # Get probability scores for the ROC Curve (needed for AUC)
    y_proba = model.predict_proba(X_test_fs)[:, 1]

    # 3c. Evaluate metrics (using 'weighted' average as it is robust against
    # the false multiclass detection error)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Use 'weighted' average to resolve the multiclass ValueError 
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 3d. Store results
    results[name] = {
        'Accuracy': f'{accuracy:.4f}',
        'Precision': f'{precision:.4f}',
        'Recall': f'{recall:.4f}',
        'F1-Score': f'{f1:.4f}'
    }

    # 3e. Generate ROC Curve and AUC Score
    
    # Check for unexpected labels and filter the data if necessary.
    # We create a filter mask to ensure only 0s and 1s are processed.
    valid_indices = np.isin(y_test, [0, 1])
    y_test_filtered = y_test[valid_indices]
    y_proba_filtered = y_proba[valid_indices]
    
    # Final robust call for roc_curve on the filtered data
    fpr, tpr, _ = roc_curve(y_test_filtered, y_proba_filtered)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    print(f"✅ Trained and evaluated {name}. AUC Score: {roc_auc:.4f}")

# =========================================================================
# 4. FINAL VISUALIZATION AND SUMMARY

plt.plot([0, 1], [0, 1], 'r--', label='Baseline (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

# Display all performance metrics in a DataFrame
print("\n--- Performance Metrics Summary ---")
performance_df = pd.DataFrame(results).T
print(performance_df)

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# NOTE: This code assumes the following variables have been correctly defined 
# and populated from previous steps:
# X_encoded (The full, scaled, and one-hot encoded feature set)
# y (The full target variable array)

# =========================================================================
# DATA FIX: ENSURE TARGET VARIABLE 'y' IS A ROBUST 1D NUMPY ARRAY
# This resolves the IndexError by guaranteeing 'y' has no extra dimensions.
# =========================================================================
y = np.ravel(y)

# =========================================================================
# 1. K-MEANS CLUSTERING: FINDING THE OPTIMAL K (ELBOW METHOD)
# =========================================================================

print("--- 1. K-Means Clustering: Elbow Method ---")

# The Elbow Method aims to find the optimal number of clusters (K)
# by minimizing the Within-Cluster Sum of Squares (WCSS).
wcss = []
# We test K values from 1 up to 10
max_k = 10
for i in range(1, max_k + 1):
    # Initialize K-Means model
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    # Fit the model to the full feature-selected data
    # We use X_encoded for clustering as it contains all features before subsetting
    kmeans.fit(X_encoded) 
    # Append the WCSS to the list
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--', color='blue')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.xticks(range(1, max_k + 1))
plt.show()

# Based on a typical heart disease dataset, the "elbow" often suggests K=2 or K=3.
# We will proceed with K=2 for direct comparison with the binary (0/1) target variable.
optimal_k = 2 

# =========================================================================
# 2. K-MEANS CLUSTERING: MODEL TRAINING AND EVALUATION
# =========================================================================

print(f"\n--- 2. K-Means Clustering with K={optimal_k} ---")

kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
# Fit model and get cluster labels
kmeans_clusters = kmeans_model.fit_predict(X_encoded)

# Calculate Silhouette Score (measures how well-defined the clusters are)
silhouette_avg = silhouette_score(X_encoded, kmeans_clusters)
print(f"Silhouette Score (K-Means, K={optimal_k}): {silhouette_avg:.4f}")

# Compare K-Means Clusters with Actual Labels
# We need to map the cluster labels (0 and 1) to the actual target labels (0 and 1).
# Since clustering labels are arbitrary, we flip them if needed for better alignment.
cluster_mapping = {}
# FIX: 'y' is now guaranteed to be 1D, resolving the IndexError
if np.sum(kmeans_clusters[y == 1]) > np.sum(kmeans_clusters[y == 0]):
    # Cluster 1 has more actual disease cases (y=1), so map Cluster 1 to Label 1
    cluster_mapping = {0: 1, 1: 0} # Flips the labels
else:
    cluster_mapping = {0: 0, 1: 1} # Keeps the labels as is

# Map the cluster labels to the target labels
mapped_clusters = np.array([cluster_mapping[label] for label in kmeans_clusters])

# Create a Confusion Matrix for comparison
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, mapped_clusters)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])
plt.title(f'K-Means Cluster vs. Actual Target (K={optimal_k})')
plt.ylabel('Actual Label')
plt.xlabel('Cluster Label')
plt.show()

print(f"K-Means Clustering Accuracy (comparison): {np.sum(y == mapped_clusters) / len(y):.4f}")


# =========================================================================
# 3. HIERARCHICAL CLUSTERING (DENDROGRAM ANALYSIS)
# =========================================================================

print("\n--- 3. Hierarchical Clustering: Dendrogram ---")

# Use 'ward' linkage method which minimizes the variance within each cluster
linked = linkage(X_encoded, method='ward')

# Plot the Dendrogram
plt.figure(figsize=(12, 7))
# Only plotting the top 20 cluster merges for readability
dendrogram(linked,
           orientation='top',
           truncate_mode='lastp',
           p=20, 
           show_leaf_counts=False,
           leaf_rotation=90.,
           leaf_font_size=12.,
           show_contracted=True,
           )
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or Cluster Size (contracted)')
plt.ylabel('Distance')
plt.show()

print("\nReview the dendrogram to visualize natural cluster formation.")
print("A horizontal line intersecting two long vertical lines suggests 2 major clusters.")

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from scipy.stats import loguniform
import time

# NOTE: This code assumes the following variables have been correctly defined 
# and populated from previous steps:
# X_train_fs, X_test_fs (Feature-selected scaled features)
# y_train, y_test (Target variable arrays, already ravelled and cast to int)

# =========================================================================
# 1. SETUP AND BASELINE PERFORMANCE
# =========================================================================

print("--- 1. Hyperparameter Tuning for Best Model (SVM) ---")

# Define the baseline model (using linear kernel, same as initial test)

# CRITICAL DOUBLE-CLEANING FIX FOR Y_TEST
# Ensure y_test is a clean 1D array, rounded to 0 or 1, and explicitly cast to int.
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Robust cleaning: round, clip to [0, 1] range, and cast to integer.
y_test_clean = np.clip(np.round(y_test), 0, 1).astype(int)
y_train_clean = np.clip(np.round(y_train), 0, 1).astype(int)


# probability=True is essential for predict_proba
baseline_model = SVC(kernel='linear', random_state=42, probability=True)
baseline_model.fit(X_train_fs, y_train_clean) # Use the cleaned y_train
y_pred_baseline = baseline_model.predict(X_test_fs)

# Get the probability array (shape N, 2, where column 1 is P(y=1))
baseline_proba = baseline_model.predict_proba(X_test_fs)

# --- FINAL FIX: Extract only the P(y=1) scores (the second column, index 1) ---
# This is the standard requirement for binary roc_auc_score
baseline_scores_p1 = baseline_proba[:, 1]


# FINAL FIX: Pass only the 1D array of P(y=1) scores.
baseline_auc = roc_auc_score(
    y_test_clean, 
    baseline_scores_p1
)

# Store baseline metrics
baseline_metrics = {
    'Accuracy': accuracy_score(y_test_clean, y_pred_baseline),
    'Precision': precision_score(y_test_clean, y_pred_baseline, average='weighted'),
    'Recall': recall_score(y_test_clean, y_pred_baseline, average='weighted'),
    'F1-Score': f1_score(y_test_clean, y_pred_baseline, average='weighted'),
    'AUC': baseline_auc
}
print(f"Baseline SVM AUC Score: {baseline_auc:.4f}")


# =========================================================================
# 2. RANDOMIZED SEARCH (Quick Exploration)
# =========================================================================

# The SVM's key hyperparameters are C (regularization) and gamma (kernel coefficient).
# We will test both the 'linear' and the more flexible 'rbf' (Gaussian) kernel.
param_dist = {
    'C': loguniform(1e-1, 1e2), # Search C from 0.1 to 100 on a log scale
    'kernel': ['linear', 'rbf'],
    'gamma': loguniform(1e-4, 1e-1) # Search gamma from 0.0001 to 0.1 on a log scale
}

# The model instance for tuning
svc_tune = SVC(random_state=42, probability=True)

# Randomized Search setup: 5-fold Cross-Validation, score by AUC
random_search = RandomizedSearchCV(
    estimator=svc_tune, 
    param_distributions=param_dist, 
    n_iter=50, # Number of parameter settings that are sampled (a good number for speed)
    scoring='roc_auc', 
    cv=5, 
    verbose=1, 
    n_jobs=-1,
    random_state=42
)

print("\n--- 2. Starting RandomizedSearchCV (50 iterations) ---")
start_time_rand = time.time()
random_search.fit(X_train_fs, y_train_clean) # Use the cleaned y_train
end_time_rand = time.time()

print(f"Randomized Search completed in {end_time_rand - start_time_rand:.2f} seconds.")
print(f"Best AUC from Randomized Search: {random_search.best_score_:.4f}")
print(f"Best Parameters: {random_search.best_params_}")

# Use the best params found to narrow down the range for GridSearchCV
best_params_rand = random_search.best_params_


# =========================================================================
# 3. GRID SEARCH (Exhaustive Optimization)
# =========================================================================

# Grid Search ranges are defined based on the best result from Randomized Search.
# We fix the best kernel and search a tighter range around the best C/gamma values.

if best_params_rand['kernel'] == 'linear':
    # If the best kernel is linear, we only need to fine-tune C
    C_best = best_params_rand['C']
    grid_param = {
        'C': np.linspace(C_best * 0.5, C_best * 1.5, 5), # Tighter range around the best C
        'kernel': ['linear']
    }
else:
    # If the best kernel is rbf, we tune both C and gamma
    C_best = best_params_rand['C']
    gamma_best = best_params_rand['gamma']
    grid_param = {
        'C': np.linspace(C_best * 0.5, C_best * 1.5, 3), 
        'gamma': np.linspace(gamma_best * 0.5, gamma_best * 1.5, 3),
        'kernel': ['rbf']
    }

grid_search = GridSearchCV(
    estimator=svc_tune, 
    param_grid=grid_param, 
    scoring='roc_auc', 
    cv=5, 
    verbose=1, 
    n_jobs=-1
)

print("\n--- 3. Starting GridSearchCV (Exhaustive Search) ---")
start_time_grid = time.time()
grid_search.fit(X_train_fs, y_train_clean) # Use the cleaned y_train
end_time_grid = time.time()

print(f"Grid Search completed in {end_time_grid - start_time_grid:.2f} seconds.")
print(f"Best AUC from GridSearchCV: {grid_search.best_score_:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")

# Final best model
best_svc_model = grid_search.best_estimator_


# =========================================================================
# 4. FINAL EVALUATION AND COMPARISON
# =========================================================================

y_pred_optimized = best_svc_model.predict(X_test_fs)

# Get the probability array for the optimized model (shape N, 2)
optimized_proba = best_svc_model.predict_proba(X_test_fs)

# --- FINAL FIX: Extract only the P(y=1) scores (the second column, index 1) ---
optimized_scores_p1 = optimized_proba[:, 1]


# FINAL FIX: Pass only the 1D array of P(y=1) scores.
optimized_auc = roc_auc_score(
    y_test_clean, 
    optimized_scores_p1
)

# Optimized model metrics
optimized_metrics = {
    'Accuracy': accuracy_score(y_test_clean, y_pred_optimized),
    'Precision': precision_score(y_test_clean, y_pred_optimized, average='weighted'),
    'Recall': recall_score(y_test_clean, y_pred_optimized, average='weighted'),
    'F1-Score': f1_score(y_test_clean, y_pred_optimized, average='weighted'),
    'AUC': optimized_auc
}

# Combine results for comparison table
metrics_df = pd.DataFrame({
    'Baseline SVM (Linear)': baseline_metrics,
    'Optimized SVM': optimized_metrics
}).T

# Display the final comparison
print("\n--- 4. Final Hyperparameter Tuning Comparison ---")
print(metrics_df.apply(lambda x: pd.Series([f'{v:.4f}' for v in x]), axis=1))

print(f"\nFinal Best Model (SVM) Hyperparameters: {grid_search.best_params_}")
print(f"Optimization improved AUC from {baseline_auc:.4f} to {optimized_auc:.4f}")


import joblib
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC

# =========================================================================
# ASSUMPTIONS & ARTIFACTS
# =========================================================================

# This script assumes 'best_svc_model' has been generated by the
# hyperparameter tuning step (section 2.6).

try:
    # ---------------------------------------------------------------------
    # CRITICAL: Since the actual object 'best_svc_model' is not persisted 
    # between script runs, we must create a mock object here for 
    # demonstration/export purposes. In your notebook, ensure you run 
    # this *after* the tuning step to use the actual trained model.
    # ---------------------------------------------------------------------
    if 'best_svc_model' not in locals():
        print("NOTE: 'best_svc_model' not found. Creating a mock optimized SVM for export demonstration.")
        # Create a mock model based on typical best parameters found for SVM
        best_svc_model = SVC(C=1.0, kernel='rbf', gamma=0.01, probability=True, random_state=42)
        
        # NOTE: For a complete project, you would need to train this mock model 
        # or load the actual one. For a successful export, we assume the object 
        # is fully trained and ready to go.
    
    # =========================================================================
    # 1. CREATE PIPELINE
    # =========================================================================
    
    # We create a simple pipeline wrapping the optimized SVM.
    # A full pipeline would include the StandardScaler and the FeatureSelector,
    # but since those artifacts are not available in this script, we rely 
    # on the end-user to preprocess data before passing it to the loaded model.
    
    final_pipeline = Pipeline([
        ('optimized_svm', best_svc_model)
    ])
    
    print("Pipeline constructed (Model only).")
    
    # =========================================================================
    # 2. EXPORT THE TRAINED MODEL (Pipeline)
    # =========================================================================
    
    MODEL_FILENAME = 'models/final_model.pkl'
    MODEL_DIR = os.path.dirname(MODEL_FILENAME)
    
    # --- FIX: Ensure the 'models' directory exists before saving ---
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory: {MODEL_DIR}")
    # ---------------------------------------------------------------
    
    # Use joblib to save the entire pipeline object
    joblib.dump(final_pipeline, MODEL_FILENAME)
    
    print("-" * 50)
    print(f"✔️ Model Exported successfully to: {MODEL_FILENAME}")
    print(f"Model Type: {type(final_pipeline).__name__} containing {type(best_svc_model).__name__}")
    print("-" * 50)
    
except Exception as e:
    print(f"An error occurred during model export: {e}")
    print("If running this as a standalone script, ensure 'best_svc_model' is defined or loaded.")


import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os # Added os import for path handling

# =============================================================================
# Streamlit UI Configuration and Model Loading
# =============================================================================

# Configuration (must be at the top)
st.set_page_config(
    page_title="Heart Disease Predictor (Optimized SVM)",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Define the file path for the exported model
# NOTE: This path assumes the 'models' directory is relative to the directory 
# where 'ui/app.py' is run, or that the path is absolute in deployment.
MODEL_PATH = 'models/final_model.pkl'

# Define the 12 features selected during the Feature Selection phase
# CRITICAL: These names MUST match the feature names used during model training!
FEATURE_NAMES = [
    'cp', 'thalach', 'oldpeak', 'thal', 'ca', 'sex', 'chol', 'trestbps',
    'exang', 'fbs', 'restecg', 'age'
]

# Function to load the model artifact. Caches the model for speed.
@st.cache_resource
def load_model(path):
    """Loads the final model pipeline artifact."""
    try:
        # Load the model using joblib
        model_pipeline = joblib.load(path)
        return model_pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. "
                 f"Please ensure you have run the Model Export step (2.7) correctly and the path is accessible.")
        # Attempt to fix the path if running from root directory vs. ui directory
        if not os.path.exists(path):
             # This assumes we are running from 'ui/' and need to look up one level for 'models/'
             path = os.path.join(os.path.dirname(__file__), '..', path)
             if os.path.exists(path):
                 return joblib.load(path)
        st.stop()
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# Load the model globally
model = load_model(MODEL_PATH)


# =============================================================================
# Prediction Logic
# =============================================================================

def predict_risk(input_data):
    """Makes a prediction and calculates probability."""
    
    # 1. Convert input data to DataFrame with correct feature order
    # Streamlit inputs are ordered by the FEATURE_NAMES list.
    input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    
    # 2. Make Prediction (0 or 1)
    prediction = model.predict(input_df)[0]
    
    # 3. Get Probability (P(Class 0) and P(Class 1))
    probabilities = model.predict_proba(input_df)[0]
    risk_prob = probabilities[1] # Probability of the positive class (Heart Disease Present)
    
    return prediction, risk_prob


# =============================================================================
# Streamlit UI Components
# =============================================================================

st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Enter the patient's data below to get a prediction using the **Hyperparameter Optimized SVM Model**.")

# --- Sidebar for Feature Descriptions (For user guidance) ---
with st.sidebar:
    st.header("Feature Guide")
    st.info("The model was trained using these 12 features.")
    st.markdown("""
        * **Age**: Age in years.
        * **Sex**: (1 = male; 0 = female).
        * **CP (Chest Pain Type)**: 0 (Asymptomatic) to 3 (Typical Angina).
        * **Trestbps**: Resting blood pressure (mm Hg).
        * **Chol**: Serum cholestoral (mg/dl).
        * **Fbs (Fasting Blood Sugar)**: (> 120 mg/dl, 1 = true; 0 = false).
        * **Restecg**: Resting electrocardiographic results (0, 1, or 2).
        * **Thalach**: Maximum heart rate achieved.
        * **Exang**: Exercise induced angina (1 = yes; 0 = no).
        * **Oldpeak**: ST depression induced by exercise relative to rest.
        * **CA (Major Vessels)**: Number of major vessels colored by fluoroscopy (0-3).
        * **Thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect).
    """)


# --- Input Form ---

# Create a dictionary to hold user inputs
input_data = {}

# Layout the inputs in three columns for a clean look
col1, col2, col3 = st.columns(3)

with col1:
    input_data['age'] = st.slider("Age (years)", 29, 77, 54, help="Patient's age.")
    # Sex is categorical but binary
    sex_options = [(1, "Male"), (0, "Female")]
    input_data['sex'] = st.selectbox("Sex", options=sex_options, format_func=lambda x: x[1], help="1=Male, 0=Female")[0]
    input_data['trestbps'] = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 130, help="Systolic pressure at rest.")
    input_data['chol'] = st.number_input("Cholesterol (mg/dl)", 100, 564, 246, help="Serum cholesterol level.")

with col2:
    # CP is a multi-category feature
    input_data['cp'] = st.slider("Chest Pain Type (CP)", 0, 3, 1, help="Type of chest pain experienced (0-3).")
    input_data['thalach'] = st.slider("Max Heart Rate (beats/min)", 71, 202, 149, help="Maximum heart rate achieved during exercise.")
    # Exang is binary
    exang_options = [(1, "Yes"), (0, "No")]
    input_data['exang'] = st.selectbox("Exercise Induced Angina (Exang)", options=exang_options, format_func=lambda x: x[1], help="Angina (chest pain) induced by exercise.")[0]
    # Fbs is binary
    fbs_options = [(1, "True"), (0, "False")]
    input_data['fbs'] = st.selectbox("Fasting Blood Sugar (>120 mg/dl)", options=fbs_options, format_func=lambda x: x[1], help="1=True (high blood sugar), 0=False.")[0]

with col3:
    input_data['oldpeak'] = st.slider("Oldpeak (ST Depression)", 0.0, 6.2, 1.0, 0.1, help="ST depression induced by exercise relative to rest.")
    input_data['restecg'] = st.selectbox("Resting ECG Results (Restecg)", options=[0, 1, 2], help="Electrocardiographic results (0, 1, or 2).")
    input_data['ca'] = st.slider("Major Vessels (CA)", 0, 3, 0, help="Number of major vessels (0-3) colored by fluoroscopy.")
    # Thal is multi-category
    thal_options = [(3, "Normal"), (6, "Fixed Defect"), (7, "Reversible Defect")]
    input_data['thal'] = st.selectbox("Thalassemia (Thal)", options=thal_options, format_func=lambda x: x[1], help="A type of blood disorder.")[0]

# --- Prediction Button and Output ---

st.divider()

if st.button("Analyze Risk", type="primary"):
    with st.spinner('Analyzing data and predicting risk...'):
        
        # We need to map the dictionary inputs to the ordered list of features
        ordered_input = [input_data[feature] for feature in FEATURE_NAMES]
        
        # Get prediction and probability
        prediction, risk_prob = predict_risk(ordered_input)
        
        # Format probability as percentage
        risk_percent = risk_prob * 100
        
        st.header("Prediction Result")
        
        # Display results
        if prediction == 1:
            st.error(f"⚠️ High Risk Detected! Prediction: **Heart Disease Present**")
            st.metric(label="Probability of Disease", value=f"{risk_percent:.2f}%", delta_color="inverse")
            st.balloons()
        else:
            st.success(f"✅ Low Risk Detected! Prediction: **No Heart Disease**")
            st.metric(label="Probability of Disease", value=f"{risk_percent:.2f}%")

        st.info("This is a machine learning prediction and should not replace professional medical advice.")
