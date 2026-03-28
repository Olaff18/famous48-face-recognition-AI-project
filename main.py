import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV


X_raw = []
y_raw = []
print("Loading and processing data from combined48.txt...")
with open('data/combined48.txt', 'r') as file:
    for line in file:
        values = line.strip().split()
        
        # lines with exactly 584 numbers 
        # (576 pixels + 8 attributes)
        if len(values) == 584:
            # convert the text strings to float numbers
            numeric_values = [float(val) for val in values]
            
            # check attribute a1 (index 576): 1 = face, 0 = no face
            if numeric_values[576] == 1.0:
                
                # extract the 576 pixels
                pixels = numeric_values[:576]
                X_raw.append(pixels)
                
                # extract attribute a3 (index 578): Person ID
                label = int(numeric_values[578])
                y_raw.append(label)

# convert lists to NumPy arrays 
X = np.array(X_raw)
y = np.array(y_raw)

print(f"Successfully loaded {X.shape[0]} valid images.")

# split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the pixels 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data is cleaned, split, and ready for modeling!")

# initialize the random forest
# 150 individual decision trees
# rf_model = RandomForestClassifier(n_estimators=150, random_state=42)

# # training
# print("Training the Random Forest model...")
# rf_model.fit(X_train, y_train)

# # testing
# print("Making predictions on the test set...")
# rf_predictions = rf_model.predict(X_test)

# # results
# accuracy = accuracy_score(y_test, rf_predictions)
# print(f"random forest accuracy: {accuracy * 100:.2f}%\n")

# # how well it recognized specific people
# print("detailed classification report:")
# print(classification_report(y_test, rf_predictions, zero_division=0))
# define the "Grid" of parameters you want to test
param_grid = {
    'n_estimators': [100, 200, 300],       # 100, 200, and 300 trees
    'max_depth': [None, 10, 20],           # unlimited depth, or capping it at 10 or 20
    'min_samples_split': [2, 5, 10]        # different splitting rules
}

# initialize a blank Random Forest
rf_base = RandomForestClassifier(random_state=42)

# set up the grid Search
# cv=3 means it will cross-validate the training data 3 times for each combination
# n_jobs=-1 tells computer to use all its processor cores to speed up
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# run the search 
print("testing 27 different combinations . . .")
grid_search.fit(X_train, y_train)

# print the winning combination
print(f"\nwinning combination: {grid_search.best_params_}")

# evaluate the optimized model on test data
best_rf = grid_search.best_estimator_
optimized_predictions = best_rf.predict(X_test)
optimized_accuracy = accuracy_score(y_test, optimized_predictions)

print(f"Optimized Random Forest Accuracy: {optimized_accuracy * 100:.2f}%\n")


# winning combination: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 300}
# optimized Random Forest Accuracy: 70.59%
