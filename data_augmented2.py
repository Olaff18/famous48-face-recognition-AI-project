import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.ndimage import shift
from sklearn.ensemble import RandomForestClassifier

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

# split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Original Training Size: {X_train.shape[0]} images")

# --- pixel shifting ---
print("Augmenting data by shifting pixels (expands the dataset by 5x)...")

X_train_augmented = list(X_train)
y_train_augmented = list(y_train)

for i in range(len(X_train)):
    # reshape to 24x24 for spatial shifting
    img_2d = X_train[i].reshape(24, 24)
    
    # create shifted versions (filling the empty edge pixels with 0, which is black)
    shifted_up = shift(img_2d, [-1, 0], cval=0.0)
    shifted_down = shift(img_2d, [1, 0], cval=0.0)
    shifted_left = shift(img_2d, [0, -1], cval=0.0)
    shifted_right = shift(img_2d, [0, 1], cval=0.0)
    
    # flatten and append them to the training pool
    X_train_augmented.extend([
        shifted_up.flatten(), 
        shifted_down.flatten(), 
        shifted_left.flatten(), 
        shifted_right.flatten()
    ])
    
    # append the same label 4 times for the 4 new shifted images
    y_train_augmented.extend([y_train[i]] * 4)

X_train = np.array(X_train_augmented)
y_train = np.array(y_train_augmented)

print(f"New Augmented Training Size: {X_train.shape[0]} images. Training will take a bit longer.")

# scale the massively expanded dataset
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


print("\n--- Training the Ultimate Neural Network ---")
ann_model_ultimate = MLPClassifier(
    hidden_layer_sizes=(512, 256),   
    activation='relu',               
    solver='adam',                   
    alpha=0.05,                      
    learning_rate='adaptive',        
    max_iter=1000,                   
    random_state=42
)

ann_model_ultimate.fit(X_train_scaled, y_train)

# make predictions
ann_ultimate_predictions = ann_model_ultimate.predict(X_test_scaled)

# calculate accuracy
ann_accuracy = accuracy_score(y_test, ann_ultimate_predictions)
print(f"Ultimate Neural Network Accuracy: {ann_accuracy * 100:.2f}%")

# new Augmented Training Size: 27340 images

# --- training Random Forest ---
# final Random Forest Accuracy: 73.74%

# --- training the Ultimate Neural Network ---
# Ultimate Neural Network Accuracy: 88.51%