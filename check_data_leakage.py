# Check for data leakage and validation set issues

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load your data splits
# Assuming X_train, X_valid, X_test, y_train, y_valid, y_test are available

print("=== DATA LEAKAGE CHECK ===")

# 1. Check for duplicate rows between train and validation
def check_duplicates(X1, X2, name1="Set1", name2="Set2"):
    # Convert to DataFrame for easier comparison
    df1 = pd.DataFrame(X1)
    df2 = pd.DataFrame(X2)
    
    # Check for exact duplicates
    duplicates = 0
    for idx, row in df2.iterrows():
        if (df1 == row).all(axis=1).any():
            duplicates += 1
    
    print(f"\nDuplicates between {name1} and {name2}: {duplicates}")
    print(f"Percentage of {name2} that exists in {name1}: {duplicates/len(df2)*100:.2f}%")
    
    return duplicates

# Check all combinations
check_duplicates(X_train, X_valid, "Train", "Validation")
check_duplicates(X_train, X_test, "Train", "Test")
check_duplicates(X_valid, X_test, "Validation", "Test")

# 2. Check similarity between sets
print("\n=== SIMILARITY CHECK ===")
def check_similarity(X1, X2, name1, name2):
    # Sample random rows to check similarity
    n_samples = min(100, len(X1), len(X2))
    idx1 = np.random.choice(len(X1), n_samples, replace=False)
    idx2 = np.random.choice(len(X2), n_samples, replace=False)
    
    similarities = cosine_similarity(X1[idx1], X2[idx2])
    avg_similarity = similarities.mean()
    max_similarity = similarities.max()
    
    print(f"\n{name1} vs {name2}:")
    print(f"Average similarity: {avg_similarity:.4f}")
    print(f"Max similarity: {max_similarity:.4f}")
    print(f"Samples with >0.99 similarity: {(similarities > 0.99).sum()}")

check_similarity(X_train, X_valid, "Train", "Validation")
check_similarity(X_train, X_test, "Train", "Test")

# 3. Check class distribution
print("\n=== CLASS DISTRIBUTION ===")
print("Train:", pd.Series(y_train).value_counts(normalize=True).sort_index())
print("Valid:", pd.Series(y_valid).value_counts(normalize=True).sort_index())
print("Test:", pd.Series(y_test).value_counts(normalize=True).sort_index())

# 4. Check if validation is too small
print("\n=== SET SIZES ===")
print(f"Train size: {len(X_train)} ({len(X_train)/(len(X_train)+len(X_valid)+len(X_test))*100:.1f}%)")
print(f"Valid size: {len(X_valid)} ({len(X_valid)/(len(X_train)+len(X_valid)+len(X_test))*100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/(len(X_train)+len(X_valid)+len(X_test))*100:.1f}%)")

# 5. Check feature variance
print("\n=== FEATURE VARIANCE ===")
train_var = X_train.var(axis=0)
valid_var = X_valid.var(axis=0)
test_var = X_test.var(axis=0)

print(f"Features with zero variance in train: {(train_var == 0).sum()}")
print(f"Features with zero variance in valid: {(valid_var == 0).sum()}")
print(f"Features with zero variance in test: {(test_var == 0).sum()}")

# 6. Recommendation
print("\n=== RECOMMENDATIONS ===")
if len(X_valid) < 500:
    print("- Validation set might be too small. Consider using cross-validation.")
    
if check_duplicates(X_train, X_valid, "", "") > 0:
    print("- Data leakage detected! Remove duplicates between train and validation.")

print("\n- Consider using stratified K-fold cross-validation for more reliable results")
print("- Try ensemble methods (Random Forest, XGBoost) as baseline")
print("- Implement data augmentation with noise injection")
print("- Use stronger regularization or simpler models")