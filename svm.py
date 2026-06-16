import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("Loading raw 576-pixel data...")

X_raw = []
y_raw = []

with open(r'famous48-face-recognition-AI-project\data\combined48.txt', 'r') as file:
    for line in file:
        values = line.strip().split()
        
        if len(values) == 584:
            numeric_values = [float(val) for val in values]
            
            # Check attribute a1 (index 576): 1 = face, 0 = no face
            if numeric_values[576] == 1.0:
                # Grab ALL 576 raw pixels (No cropping, no HOG!)
                pixels = numeric_values[:576]
                X_raw.append(pixels)
                
                # Extract label
                label = int(numeric_values[578])
                y_raw.append(label)

X = np.array(X_raw)
y = np.array(y_raw)
print(f"Successfully loaded {X.shape[0]} images.")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the raw pixels (Critical for both ANN and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 2. SVM on Raw Pixels ---
print("\n--- Training SVM on Raw Scaled Pixels ---")
svm_model_raw = SVC(
    kernel='rbf', 
    C=10, 
    gamma=0.001, 
    random_state=42
)

svm_model_raw.fit(X_train_scaled, y_train)
svm_raw_predictions = svm_model_raw.predict(X_test_scaled)
print(f"Raw Pixel SVM Accuracy: {accuracy_score(y_test, svm_raw_predictions) * 100:.2f}%")
# --- Training SVM on Raw Scaled Pixels ---
# Raw Pixel SVM Accuracy: 86.61%
