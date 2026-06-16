import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

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

# --- 1. The Upgraded Neural Network ---

# --- 2. SVM on Raw Pixels ---
print("\n--- Running Grid Search for Perfect SVM Settings ---")
print("(This might take 5-10 minutes to run, grab a coffee!)")

# 1. Define the different settings we want to test
param_grid = {
    'C': [1, 10, 50, 100],                 # Testing different strictness levels
    'gamma': ['scale', 'auto', 0.001, 0.01], # Testing different pixel influence levels
    'kernel': ['rbf']                      # We know RBF is the best shape
}

# 2. Set up the automated tester (using 3-fold cross-validation)
grid_search = GridSearchCV(
    SVC(random_state=42), 
    param_grid, 
    cv=3, 
    scoring='accuracy', 
    verbose=2, # This prints out progress so you aren't staring at a blank screen
    n_jobs=-1  # Uses all your CPU cores to make it run faster!
)

# 3. Release the hounds! 
grid_search.fit(X_train_scaled, y_train)

# 4. Print the absolute best settings it found
print(f"\nBest Parameters Found: {grid_search.best_params_}")

# 5. Test the ultimate champion model on your test data
best_svm = grid_search.best_estimator_
final_predictions = best_svm.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, final_predictions)

print(f"Ultimate Tuned SVM Accuracy: {final_accuracy * 100:.2f}%")