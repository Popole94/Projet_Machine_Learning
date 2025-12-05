#!/usr/bin/env python
# coding: utf-8

# ### Importation of libraries and dataset

# In[1]:


from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
phishing_websites = fetch_ucirepo(id=327)
X = phishing_websites.data.features
y = phishing_websites.data.targets


# ### Dataset analysis

# In[2]:


print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}\n")
print("First 5 rows of features:")
display(X.head())
print("\nDataset Info :")
print(X.info())
print("\nMissing Values :")
print(X.isna().sum().sum())
print("\nDescriptive Statistics :")
display(X.describe())
print("\nTarget Distribution :")
print(y.value_counts())

y_series = pd.Series(y.values.ravel(), name="label")

plt.figure(figsize=(6,4))
sns.countplot(x=y_series)
plt.title("Label distribution (1 = phishing, -1 = legitimate)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

#corelation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(X.corr(), cmap="coolwarm", center=0)
plt.title("Correlation heatmap features")
plt.show()


# ### Duplicate Analysis
# 
# This section analyzes duplicate rows in the phishing websites dataset.
# 
# Important note on duplicates:
# This dataset uses discrete features with values limited to {-1, 0, 1}, representing boolean 
# properties of websites (for example: presence of IP address, SSL state, etc.). With 30 binary or 
# ternary features, the number of unique possible combinations is naturally limited. 
# 
# As a result, a high duplicate rate (>70%) is expected and normal for this dataset. 
# Phishing websites often share the same technical signatures, so many rows will have identical 
# feature values.
# 
# What matters to check:
# 1. Labeling conflicts (same features, different targets) - indicates a real problem
# 2. Class distribution (balance between phishing and legitimate sites)
# 3. Overall data consistency
# 

# In[3]:


# Check for duplicates in the dataset
data_complete = pd.concat([X, y], axis=1)

# Complete duplicates (same features and target)
complete_duplicates = data_complete.duplicated(keep=False).sum()
print(f"\nComplete duplicates: {complete_duplicates} ({complete_duplicates/len(data_complete)*100:.1f}%)")

# Feature duplicates (ignoring target)
feature_duplicates = X.duplicated(keep=False).sum()
unique_combinations = X.drop_duplicates().shape[0]
print(f"Feature duplicates: {feature_duplicates} ({feature_duplicates/len(X)*100:.1f}%)")
print(f"Unique feature combinations: {unique_combinations}")

# Check for labeling conflicts (same features, different labels)
print("\nChecking for labeling conflicts :")
conflicts = 0
conflict_rows = 0

if feature_duplicates > 0:
    for features in X[X.duplicated(keep=False)].drop_duplicates().values:
        mask = (X == features).all(axis=1)
        labels = y.loc[mask].iloc[:, 0].unique()
        if len(labels) > 1:
            conflicts += 1
            conflict_rows += mask.sum()

print(f"Found {conflicts} conflicting feature groups")
if conflicts > 0:
    print(f"Total rows with conflicts: {conflict_rows} ({conflict_rows/len(X)*100:.1f}%)")
    print("Note: These should be investigated as they have the same features but different labels")

# Feature characteristics
print("\nFeature characteristics:")
cardinality = X.nunique()
print(f"Binary features (2 values): {(cardinality == 2).sum()}")
print(f"Ternary features (3 values): {(cardinality == 3).sum()}")


# Summary:
# The high duplicate rate is normal for this dataset since features are discrete (values -1, 0, 1). Many phishing sites share similar characteristics, which explains why 5785 unique combinations appear 1.9 times on average.
# 

# ### Outlier Analysis
# 
# This section detects and analyzes outliers in the dataset using Isolation Forest.

# In[4]:


# Outlier Analysis using Isolation Forest
from sklearn.ensemble import IsolationForest


# Detect outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X)
outlier_count = (outliers == -1).sum()

print(f"\nOutliers detected: {outlier_count} ({outlier_count/len(X)*100:.1f}%)")
print("(Using Isolation Forest with 5% contamination)")

# Check class distribution
print("\nClass distribution:")
print(y.iloc[:, 0].value_counts().sort_index())
print()
print(y.iloc[:, 0].value_counts(normalize=True).sort_index() * 100)

# Calculate class imbalance
class_counts = y.iloc[:, 0].value_counts()
if len(class_counts) == 2:
    ratio = class_counts.iloc[0] / class_counts.iloc[1]
    print(f"\nClass imbalance ratio: {ratio:.2f}:1")

# Check outliers per class
print("\nOutliers by class:")
for class_label in y.iloc[:, 0].unique():
    class_mask = y.iloc[:, 0] == class_label
    class_outliers = (outliers[class_mask] == -1).sum()
    total_class = class_mask.sum()
    print(f"Class {class_label}: {class_outliers} outliers ({class_outliers/total_class*100:.1f}%)")



# We detected 553 outliers (5%) using Isolation Forest, which is expected since we set the contamination parameter to 0.05. These represent websites with unusual or rare feature combinations.
# The dataset has good class balance with 55.7% legitimate sites and 44.3% phishing sites, giving a class imbalance ratio of 1.26:1 which is reasonable for classification tasks.
# 
# One interesting finding is that phishing sites have more outliers (7.2%) compared to legitimate sites (3.2%). This makes sense because phishing websites use a wider variety of techniques and attack methods, while legitimate sites tend to follow more standard patterns. This is actually a good sign that our dataset captures the real-world complexity of the problem.
# 
# Overall, the data is well-balanced and ready for model training without requiring special handling for class imbalance.

# ### Overall Dataset Quality Assessment
# 
# Based on the comprehensive analysis of duplicates, outliers, and class distribution, this dataset demonstrates good quality for machine learning:
# 
# **Strengths:**
# - No missing values in the dataset
# - Well-balanced classes (1.26:1 ratio) - excellent for classification
# - Sufficient sample size (11055 records)
# - Discrete features are properly encoded ({-1, 0, 1}) and consistent
# - 22 binary features and 8 ternary features provide solid dimensionality
# - Outlier distribution is reasonable and logically explained
# 
# **Areas of Attention:**
# - 64 labeling conflicts (3.2% of data) should be investigated before final model deployment
# - High duplicate rate (70.9%) is normal for this domain but means the effective unique patterns are limited to ~5785 combinations
# - Model generalization may be affected by the limited feature space diversity
# 
# **Recommendations:**
# 1. Keep all records including duplicates - they represent real-world patterns
# 2. Consider investigating the 64 conflicting cases to understand if they represent true ambiguities or labeling errors
# 3. Use stratified cross-validation during model training to preserve class distribution
# 4. Monitor model performance on both phishing and legitimate site classes separately
# 5. The dataset is ready for training with standard approaches; no special preprocessing required
# 
# **Overall Rating: GOOD** - The dataset is suitable for building a phishing detection classifier with expected performance. While the high duplicate rate initially appears concerning, it is characteristic of discrete feature datasets and does not indicate data quality issues.

# ### Data Pre-processing
# 
# Cette section prépare les données pour l'entraînement du modèle. Nous effectuons les transformations suivantes:
# 1. **Encodage de la target**: Conversion de (-1, 1) à (0, 1) pour compatibilité avec les métriques standard
# 2. **Normalisation**: Application du StandardScaler pour mettre à l'échelle les features
# 3. **Train-test split**: Division stratifiée pour maintenir l'équilibre des classes
# 

# In[5]:


#Encode target (-1,1) to (0,1) 
y_binary = y.replace({-1: 0, 1: 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
)

#standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nTrain set:", X_train.shape, " Test set:", X_test.shape)


# ### Creation of the first models + hyperparameter tuning
# 
# Dans cette section, nous construisons des modèles de baseline et les optimisons via tuning d'hyperparamètres.
# Nous utilisons RandomForest comme modèle initial, qui est robuste et fournit une bonne performance de base.
# Les hyperparamètres sont ajustés pour améliorer la précision et réduire l'overfitting.

# In[6]:


print("\nTraining baseline model (Random Forest)")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
#evaluation
print("\nBaseline Model Evaluation :")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report :\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of Random Forest (Baseline)")
plt.show()


# ## GridSearch

# In[7]:


from sklearn.tree import DecisionTreeClassifier, plot_tree


# Create the base model
base_dt = DecisionTreeClassifier(random_state=42)
base_dt.fit(X_train, y_train)
print("\nBaseline test accuracy:", accuracy_score(y_test, base_dt.predict(X_test)))


# In[8]:


# Define hyperparameter grid

print("\n Defining hyperparameter grid...")

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None]  # Important for imbalanced data!
}

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\n Hyperparameter grid defined:")
print(f"  - Total combinations: {total_combinations}")
print(f"  - Parameters to tune: {list(param_grid.keys())}")


# In[9]:


from sklearn.model_selection import GridSearchCV

# Perform GridSearchCV
print("\nRunning GridSearchCV (this may take a few minutes)...")

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1',  # Better metric for imbalanced data
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all CPU cores
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train.values.ravel())

print(f"  - Best CV F1-Score: {grid_search.best_score_:.4f}")
print(f"  - Best parameters: {grid_search.best_params_}")


# In[10]:


# Visualize optimal tree
plt.figure(figsize=(18, 10))
plot_tree(
    grid_search.best_estimator_,
    filled=True,
    feature_names=phishing_websites.feature_names,
    class_names=phishing_websites.target_names,
    rounded=True
)
plt.show()


# ### Address overfitting / underfitting

# In[ ]:





# ### Relevant metrics

# In[ ]:





# ### Dimension reduction

# In[ ]:





# ### Ensemble models + advanced models

# In[ ]:





# ### Comparison of models + Conclusion
