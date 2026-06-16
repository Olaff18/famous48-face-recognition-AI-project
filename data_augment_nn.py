import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading raw 576-pixel data...")

X_raw = []
y_raw = []

with open(r'famous48-face-recognition-AI-project\data\combined48.txt', 'r') as file:
    for line in file:
        values = line.strip().split()
        if len(values) == 584:
            numeric_values = [float(val) for val in values]
            # Check attribute a1: 1 = face
            if numeric_values[576] == 1.0:
                X_raw.append(numeric_values[:576])
                y_raw.append(int(numeric_values[578]))

X = np.array(X_raw)
y = np.array(y_raw)
print(f"Successfully loaded {X.shape[0]} images.")

# 1. split the data first then augment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Original Training Size: {X_train.shape[0]} images")

# --- DATA AUGMENTATION (horizontal flipping) ---
print("Augmenting data... (Flipping images horizontally)")
X_train_augmented = list(X_train)
y_train_augmented = list(y_train)

for i in range(len(X_train)):
    # reshape the flat 576 array back into a 24x24 picture
    img_2d = X_train[i].reshape(24, 24)
    
    # flip it like a mirror
    img_flipped = np.fliplr(img_2d)
    
    # flatten it back to 576 and add it to our training pool
    X_train_augmented.append(img_flipped.flatten())
    y_train_augmented.append(y_train[i]) # Add the same label!

# convert back to numpy arrays for the model
X_train = np.array(X_train_augmented)
y_train = np.array(y_train_augmented)

print(f"New Augmented Training Size: {X_train.shape[0]} images!")

# 2. scale the newly massive dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\n--- Training the Ultimate Neural Network ---")

ann_model_ultimate = MLPClassifier(
    hidden_layer_sizes=(512, 256),   # sweet spot for 24x24 images
    activation='relu',               # standard, fast activation
    solver='adam',                   # best overall optimizer
    alpha=0.05,                      # stronger regularization to prevent overfitting
    learning_rate='adaptive',        # shifts gears to make fine-tuned adjustments
    max_iter=1000,                   # let it cook
    random_state=42
)


# neural networks need the scaled data
ann_model_ultimate.fit(X_train_scaled, y_train)

# make predictions
ann_ultimate_predictions = ann_model_ultimate.predict(X_test_scaled)

# calculate accuracy
ann_accuracy = accuracy_score(y_test, ann_ultimate_predictions)
print(f" Ultimate Neural Network Accuracy: {ann_accuracy * 100:.2f}%")

print("\nDetailed Report:")
print(classification_report(y_test, ann_ultimate_predictions, zero_division=0))

# new Augmented training size: 10936 images

# --- training the Ultimate Neural Network ---
#  Ultimate Neural Network Accuracy: 86.17%