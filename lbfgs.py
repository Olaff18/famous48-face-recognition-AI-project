from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


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




print("\n--- Running Grid Search for the Perfect Neural Network ---")
print("(Grab a coffee, training multiple brain architectures takes time!)")

# 1. Define the brain architectures we want to test
param_grid = {
    'hidden_layer_sizes': [(512, 256), (512, 128)], # Removed massive layers to speed up
    'solver': ['lbfgs'],                            # Removed slow 'adam'
    'alpha': [0.01, 0.05, 0.1],                     # Testing different overfitting penalties
    'activation': ['relu', 'tanh']                  # 'tanh' sometimes handles pixel data beautifully
}

# 2. Set up the automated tester
nn_grid_search = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42), 
    param_grid, 
    cv=3, 
    scoring='accuracy', 
    verbose=2, 
    n_jobs=-1  # Use all CPU cores!
)

# 3. Release the hounds!
nn_grid_search.fit(X_train_scaled, y_train)

# 4. Print the absolute best settings
print("\n--- All Grid Search Results ---")
results = nn_grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(f"Accuracy: {mean_score * 100:.2f}% | Parameters: {params}")

print(f"\nBest NN Parameters Found: {nn_grid_search.best_params_}")

# --- THE FAIL-SAFE: Save the model ---
best_nn = nn_grid_search.best_estimator_
joblib.dump(best_nn, 'champion_neural_network.pkl')
print("Model saved to disk as 'champion_neural_network.pkl'")

# 5. Test the ultimate champion model on your test data
final_predictions = best_nn.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, final_predictions)

print(f"\nUltimate Tuned NN Accuracy: {final_accuracy * 100:.2f}%")

# Best NN Parameters Found: {'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (512, 256), 'solver': 'lbfgs'}
# Model saved to disk as 'champion_neural_network.pkl'

# Ultimate Tuned NN Accuracy: 83.69%