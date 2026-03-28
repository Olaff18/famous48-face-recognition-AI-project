import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


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
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)

# training
print("Training the Random Forest model...")
rf_model.fit(X_train, y_train)

# testing
print("Making predictions on the test set...")
rf_predictions = rf_model.predict(X_test)

# results
accuracy = accuracy_score(y_test, rf_predictions)
print(f"random forest accuracy: {accuracy * 100:.2f}%\n")

# how well it recognized specific people
print("detailed classification report:")
print(classification_report(y_test, rf_predictions, zero_division=0))