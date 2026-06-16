import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

print("Loading raw 576-pixel data...")

X_raw = []
y_raw = []

with open(r'famous48-face-recognition-AI-project\data\combined48.txt', 'r') as file:
    for line in file:
        values = line.strip().split()
        if len(values) == 584:
            numeric_values = [float(val) for val in values]
            # check attribute a1: 1 = face
            if numeric_values[576] == 1.0:
                X_raw.append(numeric_values[:576])
                y_raw.append(int(numeric_values[578]))

X = np.array(X_raw)
y = np.array(y_raw)
print(f"Successfully loaded {X.shape[0]} images.")

# 1. split the data first then augment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Original Training Size: {X_train.shape[0]} images")

# --- DATA AUGMENTATION (horiz flip) ---
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
    y_train_augmented.append(y_train[i]) # add the same label!

# convert back to numpy arrays for the model
X_train = np.array(X_train_augmented)
y_train = np.array(y_train_augmented)

print(f"New Augmented Training Size: {X_train.shape[0]} images!")

# 2. scale the newly massive dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=300, 
    max_depth=None, 
    min_samples_split=2, 
    random_state=42
)


rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Final Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# 3. train our Champion Tuned SVM
print("\n--- Training Champion Model (Tuned SVM) on Augmented Data ---")
champion_model = SVC(
    kernel='rbf', 
    C=10, 
    gamma=0.001, 
    random_state=42
)

champion_model.fit(X_train_scaled, y_train)
final_predictions = champion_model.predict(X_test_scaled)

print(f"FINAL AUGMENTED ACCURACY: {accuracy_score(y_test, final_predictions) * 100:.2f}%")

# --- training Champion Model (Tuned SVM) on Augmented Data ---
# FINAL AUGMENTED ACCURACY: 86.54%