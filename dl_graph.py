import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.ndimage import shift
import copy

print("Loading data for PyTorch CNN...")

X_raw = []
y_raw = []

with open(r'famous48-face-recognition-AI-project\data\combined48.txt', 'r') as file:
    for line in file:
        values = line.strip().split()
        if len(values) == 584:
            numeric_values = [float(val) for val in values]
            if numeric_values[576] == 1.0:
                X_raw.append(numeric_values[:576])
                y_raw.append(int(numeric_values[578]))

X = np.array(X_raw)
y = np.array(y_raw)
print(f"Successfully loaded {X.shape[0]} images.")

# 1. labels strictly 0 to (num_classes - 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# 2. split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

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

# 4. reshape into 2D matrices for the CNN 
# shape requirement: (Number of Images, Channels, Height, Width)
X_train_cnn = X_train_scaled.reshape(-1, 1, 24, 24)
X_test_cnn = X_test_scaled.reshape(-1, 1, 24, 24)

# 5. convert Numpy arrays to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_cnn, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_cnn, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 6. create DataLoaders (feeding data in batches of 64)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# ---  CNN ARCHITECTURE ---
class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceCNN, self).__init__()
        
        # layer 1: The first magnifying glass (Edge detection)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks 24x24 to 12x12
        
        # layer 2: Finding more complex shapes (Eyes, Noses)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks 12x12 to 6x6
        
        # layer 3: The standard Neural Network brain on top
        self.flatten = nn.Flatten()
        
        # 64 channels * 6 height * 6 width = 2304 flat features
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.relu3 = nn.ReLU()
        
        # dropout: randomly turns off 50% of neurons during training to prevent memorization
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # pass data through the layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


# --- Setup Training Environment ---
# Automatically use your graphics card if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- Training CNN on: {device} ---")

model = FaceCNN(num_classes).to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import matplotlib.pyplot as plt

# create empty lists to store our data points for the graph
# create empty lists to store our data points for the graph
train_losses = []
val_accuracies = []

# --- variables to track the highest score ---
best_acc = 0.0
best_model_weights = None

epochs = 30
for epoch in range(epochs):
    model.train() # Training mode
    running_loss = 0.0
    
    # --- TRAINING PHASE ---
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()               
        outputs = model(inputs)             
        loss = criterion(outputs, labels)   
        loss.backward()                     
        optimizer.step()                    
        
        running_loss += loss.item()
    
    # calculate the average loss for this epoch
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    # --- VALIDATION PHASE (checking accuracy at the end of every epoch) ---
    model.eval() # turn off dropout for testing
    correct = 0
    total = 0
    
    with torch.no_grad():
        # Feed the test data in
        test_inputs = X_test_tensor.to(device)
        test_labels = y_test_tensor.to(device)
        test_outputs = model(test_inputs)
        
        # see how many it got right
        _, predicted = torch.max(test_outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()
        
    epoch_val_acc = (correct / total) * 100
    val_accuracies.append(epoch_val_acc)
    
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_val_acc:.2f}%")
    
    # --- snapshot the brain if it beat the high score! ---
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        # Deepcopy saves the exact internal dials at this specific moment
        best_model_weights = copy.deepcopy(model.state_dict()) 

# --- load the absolute best brain back into the model ---
print(f"\nTraining complete! Reloading the best brain... ")
model.load_state_dict(best_model_weights)
print(f"LOCKED IN ULTIMATE ACCURACY: {best_acc:.2f}%")

# --- DRAW THE GRAPH ---
plt.figure(figsize=(10, 5))

# plot 1: Training Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='red')
plt.title('Training Loss over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plot 2: Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='blue')
# Add a line showing exactly where the highest score was captured
plt.axhline(y=best_acc, color='green', linestyle='--', label=f'Best: {best_acc:.2f}%')
plt.title('Validation Accuracy over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()