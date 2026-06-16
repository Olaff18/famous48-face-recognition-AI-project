import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


print("Loading and preprocessing data...")

X_raw = []
y_raw = []

with open(r'famous48-face-recognition-AI-project\data\combined48.txt', 'r') as file:
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


print("\n--- Training the Ultimate Neural Network ---")

ann_model_ultimate = MLPClassifier(
    hidden_layer_sizes=(512, 256),   # the sweet spot for 24x24 images
    activation='relu',               # standard, fast activation
    solver='adam',                   # best overall optimizer
    alpha=0.05,                      # stronger regularization to prevent overfitting
    learning_rate='adaptive',        # shifts gears to make fine-tuned adjustments
    max_iter=1000,                  
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

# --- Training the Ultimate Neural Network ---
#  Ultimate Neural Network Accuracy: 85.88%