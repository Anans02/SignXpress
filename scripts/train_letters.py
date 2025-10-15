# scripts/train_letters.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import matplotlib.pyplot as plt
import os

print("🧠 TRAINING ALPHABET MODEL (A-Z ONLY)")

# Create models folder if not exists
os.makedirs('models', exist_ok=True)

# Load letter data
try:
    with open('datasets/letters/letter_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X = data['landmarks']
    y = data['labels']
    
    print(f"✅ Letter data loaded: {X.shape[0]} samples")
    print(f"🎯 Classes: {sorted(set(y))}")
    print(f"📊 Number of classes: {len(set(y))}")
    
except FileNotFoundError:
    print("❌ Letter data not found! Run process_letters.py first")
    exit()

# Encode labels (A-Z only)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n🔢 Label encoding:")
for i, letter in enumerate(label_encoder.classes_):
    print(f"   {letter} -> {i}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n📚 Data Split:")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing: {X_test.shape[0]} samples")

# Build alphabet model
def create_alphabet_model():
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(X.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(26, activation='softmax')  # 26 letters A-Z
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train model
model = create_alphabet_model()

print("\n🏗️ Model Architecture:")
model.summary()

print("\n🚀 Training alphabet model...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=0.0001)
    ]
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 ALPHABET MODEL TEST ACCURACY: {test_accuracy*100:.2f}%")

# Save everything
model.save('models/alphabet_model.h5')
with open('models/letter_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("💾 Alphabet model saved: models/alphabet_model.h5")
print("💾 Label encoder saved: models/letter_label_encoder.pkl")

# Detailed report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📈 ALPHABET CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Alphabet Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Alphabet Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/alphabet_training_history.png', dpi=300)
print("📊 Training plot saved: models/alphabet_training_history.png")

print("\n🎉 ALPHABET MODEL TRAINING COMPLETE!")
if test_accuracy >= 0.90:
    print("🔥 EXCELLENT! Your alphabet model is ready!")
elif test_accuracy >= 0.85:
    print("✅ VERY GOOD! Your alphabet model will work well!")
else:
    print("⚠️ Good! Your alphabet model is ready for testing!")