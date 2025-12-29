import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import pickle

# Load and prepare data
df = pd.read_csv("final-19-12.csv")

# Sample data as in original
malicious_df = df[df['Final Classification'] == "Malicious"]
No_Action_df = df[df['Final Classification'] == "No Action"]
Spam_df = df[df['Final Classification'] == "Spam"]

malicious_sampled = malicious_df.sample(n=len(malicious_df)-5500, random_state=42)
No_Action_sampled = No_Action_df.sample(n=len(No_Action_df)-300, random_state=42)
Spam_sampled = Spam_df.sample(n=len(Spam_df)-300, random_state=42)

df = pd.concat([malicious_sampled, No_Action_sampled, Spam_sampled], ignore_index=True)

# Load label encoders
with open("le_spf_result.pkl", "rb") as f:
    le_spf_result = pickle.load(f)
with open("le_request_type.pkl", "rb") as f:
    le_request_type = pickle.load(f)
with open("le_dkim_result.pkl", "rb") as f:
    le_dkim_result = pickle.load(f)
with open("le_dmarc_result.pkl", "rb") as f:
    le_dmarc_result = pickle.load(f)
with open("le_tls_version.pkl", "rb") as f:
    le_tls_version = pickle.load(f)
with open("le_ssl_validity_status.pkl", "rb") as f:
    le_ssl_validity_status = pickle.load(f)
with open("le_unique_parent_process_names.pkl", "rb") as f:
    le_unique_parent_process_names = pickle.load(f)

# Apply encodings
df["spf_result"] = le_spf_result.transform(df["spf_result"])
df["request_type"] = le_request_type.transform(df["request_type"])
df["dkim_result"] = le_dkim_result.transform(df["dkim_result"])
df["dmarc_result"] = le_dmarc_result.transform(df["dmarc_result"])
df["tls_version"] = le_tls_version.transform(df["tls_version"])
df["ssl_validity_status"] = le_ssl_validity_status.transform(df["ssl_validity_status"])
df["unique_parent_process_names"] = le_unique_parent_process_names.transform(df["unique_parent_process_names"])

# Prepare X and y
X = df.drop(["Final Classification"], axis=1)
y = df["Final Classification"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=47, stratify=y_encoded)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_valid.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Use StandardScaler instead of MinMaxScaler (better for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open("scaler_improved.pkl", "wb") as f:
    pickle.dump(scaler, f)

def create_improved_model(input_dim):
    """
    Simplified model architecture to reduce overfitting
    """
    model = Sequential()
    
    # Input layer with moderate neurons
    model.add(Dense(32, activation='relu', input_shape=(input_dim,),
                   kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Hidden layer
    model.add(Dense(16, activation='relu',
                   kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(3, activation='softmax'))
    
    return model

# Create model
model = create_improved_model(X_train_scaled.shape[1])

# Compile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model_improved.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train model with callbacks and smaller batch size
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_valid_scaled, y_valid),
    epochs=100,
    batch_size=32,  # Smaller batch size
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Evaluate model
y_pred_train = model.predict(X_train_scaled).argmax(axis=-1)
y_pred_valid = model.predict(X_valid_scaled).argmax(axis=-1)
y_pred_test = model.predict(X_test_scaled).argmax(axis=-1)

print("\n=== IMPROVED MODEL RESULTS ===")
print(f"\nTraining Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Validation Accuracy: {accuracy_score(y_valid, y_pred_valid):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_valid, y_pred_valid, target_names=label_encoder.classes_))

print("\nTest Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy - Improved')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss - Improved')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('improved_model_history.png')
plt.close()

# Save final model
model.save('ann_model_improved_final.h5')

# Additional technique: Class weight balancing
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print("\nClass weights for balanced training:")
for i, weight in class_weight_dict.items():
    print(f"Class {label_encoder.classes_[i]}: {weight:.4f}")

# Retrain with class weights
model_balanced = create_improved_model(X_train_scaled.shape[1])
model_balanced.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_balanced = model_balanced.fit(
    X_train_scaled, y_train,
    validation_data=(X_valid_scaled, y_valid),
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save balanced model
model_balanced.save('ann_model_balanced.h5')

print("\nImprovement strategies implemented:")
print("1. Simplified architecture (2 hidden layers instead of 6)")
print("2. Reduced neurons per layer (32->16 instead of 40->50->80->65->30)")
print("3. Lower, consistent dropout rate (0.3)")
print("4. Stronger regularization (L1+L2)")
print("5. StandardScaler instead of MinMaxScaler")
print("6. No SVD dimensionality reduction")
print("7. Early stopping and learning rate reduction")
print("8. Smaller batch size (32 instead of 800)")
print("9. Class weight balancing for imbalanced data")