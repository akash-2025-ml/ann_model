# Run this in your Jupyter notebook environment

# Cell 1: Improved Model Implementation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import pickle

# Cell 2: Load data (use your existing code)
df = pd.read_csv("final-19-12.csv")

# Sample data as before
malicious_df = df[df['Final Classification'] == "Malicious"]
No_Action_df = df[df['Final Classification'] == "No Action"]  
Spam_df = df[df['Final Classification'] == "Spam"]

malicious_sampled = malicious_df.sample(n=len(malicious_df)-5500, random_state=42)
No_Action_sampled = No_Action_df.sample(n=len(No_Action_df)-300, random_state=42)
Spam_sampled = Spam_df.sample(n=len(Spam_df)-300, random_state=42)

df = pd.concat([malicious_sampled, No_Action_sampled, Spam_sampled], ignore_index=True)

# Cell 3: Apply existing label encoders
# Load your existing encoders
with open("le_spf_result.pkl", "rb") as f:
    le_spf_result = pickle.load(f)
# ... (load all other encoders as in your code)

# Apply encodings
df["spf_result"] = le_spf_result.fit_transform(df["spf_result"])
# ... (apply all other encodings)

# Cell 4: Prepare data WITHOUT SVD
X = df.drop(["Final Classification"], axis=1)
y = df["Final Classification"]

# Load your existing label encoder
with open("label_encoder-output.pkl", "rb") as f:
    label_encoder = pickle.load(f)

y_encoded = label_encoder.transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=47, stratify=y_encoded)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Cell 5: Use StandardScaler instead of MinMaxScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Cell 6: Simplified Model Architecture
model = Sequential()

# Much simpler architecture
model.add(Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],),
               kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(16, activation='relu',
               kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(3, activation='softmax'))

# Cell 7: Compile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Cell 8: Define callbacks
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

# Cell 9: Train with smaller batch size
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_valid_scaled, y_valid),
    epochs=100,
    batch_size=32,  # Much smaller than 800
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Cell 10: Evaluate
y_pred_train = model.predict(X_train_scaled).argmax(axis=-1)
y_pred_valid = model.predict(X_valid_scaled).argmax(axis=-1) 
y_pred_test = model.predict(X_test_scaled).argmax(axis=-1)

print("=== IMPROVED MODEL RESULTS ===")
print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Validation Accuracy: {accuracy_score(y_valid, y_pred_valid):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")

# Cell 11: Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy - Improved')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss - Improved')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Cell 12: Save improved model
model.save('ann_model_improved.h5')
with open("scaler_improved.pkl", "wb") as f:
    pickle.dump(scaler, f)