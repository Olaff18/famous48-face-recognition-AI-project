import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


print("Loading and preprocessing data...")

X_raw = []
y_raw = []

with open('data/combined48.txt', 'r') as file:
    for line in file:
        values = line.strip().split()
        
        if len(values) == 584:
            numeric_values = [float(val) for val in values]
            
            # check attribute a1 (index 576): 1 = face, 0 = no face
            if numeric_values[576] == 1.0:
                pixels = numeric_values[:576]
                X_raw.append(pixels)
                
                # extract attribute a3 (index 578): Person ID
                label = int(numeric_values[578])
                y_raw.append(label)

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