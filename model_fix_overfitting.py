# More aggressive overfitting prevention

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import pickle

# Custom callback to monitor overfitting
class OverfittingMonitor(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.15):
        self.threshold = threshold
        
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        
        if train_acc - val_acc > self.threshold:
            print(f"\nWarning: Overfitting detected! Gap: {train_acc - val_acc:.4f}")

# Load your preprocessed data
# Assuming df, X_train, X_valid, X_test are already prepared

# CRITICAL: Ensure proper data split with no leakage
print("Data distribution check:")
print("Training class distribution:", pd.Series(y_train).value_counts(normalize=True))
print("Validation class distribution:", pd.Series(y_valid).value_counts(normalize=True))
print("Test class distribution:", pd.Series(y_test).value_counts(normalize=True))

# Much simpler architecture
def create_minimal_model(input_dim):
    K.clear_session()
    
    model = Sequential([
        # Add noise to inputs to prevent memorization
        GaussianNoise(0.1),
        
        # Single hidden layer with heavy regularization
        Dense(16, activation='relu', 
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
              input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.5),  # Higher dropout
        
        # Output layer
        Dense(3, activation='softmax',
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01))
    ])
    
    return model

# Alternative: Even simpler linear model
def create_linear_model(input_dim):
    model = Sequential([
        Dense(3, activation='softmax', input_shape=(input_dim,))
    ])
    return model

# Data augmentation for tabular data
def augment_data(X, y, noise_level=0.05):
    """Add noise to create more training samples"""
    X_aug = X + np.random.normal(0, noise_level, X.shape)
    return np.vstack([X, X_aug]), np.hstack([y, y])

# Augment training data
X_train_aug, y_train_aug = augment_data(X_train_scaled, y_train)
print(f"Augmented training samples: {X_train_aug.shape[0]}")

# Create and compile model
model = create_minimal_model(X_train_scaled.shape[1])

model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Even lower learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# More aggressive early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Stop earlier
    restore_best_weights=True,
    verbose=1,
    mode='min'
)

# Train with augmented data
history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_valid_scaled, y_valid),
    epochs=50,  # Fewer epochs
    batch_size=128,  # Larger batch size
    callbacks=[early_stopping, OverfittingMonitor()],
    verbose=1
)

# Evaluate
print("\n=== MINIMAL MODEL RESULTS ===")
y_pred_train = model.predict(X_train_scaled).argmax(axis=-1)
y_pred_valid = model.predict(X_valid_scaled).argmax(axis=-1)
y_pred_test = model.predict(X_test_scaled).argmax(axis=-1)

print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Validation Accuracy: {accuracy_score(y_valid, y_pred_valid):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")

# Check if validation is too easy
print("\nValidation set analysis:")
print("Unique validation samples:", X_valid_scaled.shape[0])
print("Validation predictions distribution:", pd.Series(y_pred_valid).value_counts())

# Try cross-validation to get better estimate
from sklearn.model_selection import StratifiedKFold

def evaluate_with_cv(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]
        
        # Scale data
        scaler_cv = StandardScaler()
        X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train)
        X_cv_val_scaled = scaler_cv.transform(X_cv_val)
        
        # Train model
        model_cv = create_minimal_model(X_cv_train_scaled.shape[1])
        model_cv.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model_cv.fit(
            X_cv_train_scaled, y_cv_train,
            epochs=30,
            batch_size=128,
            verbose=0
        )
        
        # Evaluate
        y_pred_cv = model_cv.predict(X_cv_val_scaled).argmax(axis=-1)
        score = accuracy_score(y_cv_val, y_pred_cv)
        cv_scores.append(score)
    
    return cv_scores

# Run cross-validation
print("\n=== CROSS-VALIDATION RESULTS ===")
X_full = np.vstack([X_train, X_valid])
y_full = np.hstack([y_train, y_valid])

cv_scores = evaluate_with_cv(X_full, y_full)
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Save the best performing model
model.save('ann_model_minimal.h5')

# Additional diagnostics
print("\n=== DIAGNOSTICS ===")
print("1. Check for class imbalance:")
print(pd.Series(y_train).value_counts())

print("\n2. Feature importance (approximate):")
# Get weights from first layer
weights = model.layers[1].get_weights()[0]  # Skip GaussianNoise layer
feature_importance = np.abs(weights).mean(axis=1)
top_features = np.argsort(feature_importance)[-10:]
print(f"Top 10 most important feature indices: {top_features}")

print("\n3. Per-class performance:")
print("\nValidation Report:")
print(classification_report(y_valid, y_pred_valid))