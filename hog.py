import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from skimage.feature import hog # IMPORT HOG
from sklearn.svm import SVC

# Notice we are using the UN-SCALED data (X_train, not X_train_scaled)
# because HOG features are already normalized!

print("Loading data and extracting HOG features...")

X_raw = []
y_raw = []

with open(r'famous48-face-recognition-AI-project\data\combined48.txt', 'r') as file:
    for line in file:
        values = line.strip().split()
        
        if len(values) == 584:
            numeric_values = [float(val) for val in values]
            
            if numeric_values[576] == 1.0:
                # 1. Grab raw pixels and reshape to 24x24
                pixels = np.array(numeric_values[:576])
                image_2d = pixels.reshape((24, 24))
                
                # 2. Apply HOG Feature Extraction
                # pixels_per_cell=(4,4) is great for tiny 24x24 images
                hog_features = hog(
                    image_2d, 
                    orientations=8, 
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2), 
                    visualize=False
                )
                
                # 3. Append the HOG features instead of raw pixels
                X_raw.append(hog_features)
                
                label = int(numeric_values[578])
                y_raw.append(label)

X = np.array(X_raw)
y = np.array(y_raw)
print(f"Loaded {X.shape[0]} images. New HOG feature size: {X.shape[1]}")

X = np.array(X_raw)
y = np.array(y_raw)
print(f"Successfully loaded {X.shape[0]} valid images.")

# split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the data 
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

# nn backpropagation
print("\n--- Training Artificial Neural Network ---")

ann_model = MLPClassifier(
    hidden_layer_sizes=(256, 128),  
    activation='relu',              
    solver='adam',                  
    max_iter=500,                   
    random_state=42
)

# neural networks need the scaled data
ann_model.fit(X_train_scaled, y_train)
ann_predictions = ann_model.predict(X_test_scaled)

ann_accuracy = accuracy_score(y_test, ann_predictions)
print(f"Final Neural Network Accuracy: {ann_accuracy * 100:.2f}%")


print("\nRandom Forest Detailed Report:")
print(classification_report(y_test, rf_predictions, zero_division=0))

print("\nNeural Network Detailed Report:")
print(classification_report(y_test, ann_predictions, zero_division=0))

# --- training Optimized Random Forest ---
# Final Random Forest Accuracy: 70.59%

# --- Training Artificial Neural Network ---
# Final Neural Network Accuracy: 84.27%