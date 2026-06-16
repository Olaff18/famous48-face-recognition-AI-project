import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.ndimage import shift

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

# 2. split data
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
    
    #aAppend the same label 4 times for the 4 new shifted images
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


# --- CNN ARCHITECTURE ---
class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceCNN, self).__init__()
        
        #layer 1: edge detection
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks 24x24 to 12x12
        
        #layer 2: finding more complex shapes (Eyes, Noses)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks 12x12 to 6x6
        
        #layer 3: flatten
        self.flatten = nn.Flatten()
        
        # 64 channels * 6 height * 6 width = 2304 flat features
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.relu3 = nn.ReLU()
        
        # Dropout: randomly turns off 50% of neurons during training to prevent memorization
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

# --- The Training Loop ---
epochs = 30
for epoch in range(epochs):
    model.train() # Set to training mode
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()               # 1. clear old math
        outputs = model(inputs)             # 2. predict
        loss = criterion(outputs, labels)   # 3. calculate how wrong it was
        loss.backward()                     # 4. learn from the mistakes
        optimizer.step()                    # 5. update the internal weights
        
        running_loss += loss.item()
        
    # print progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f}; Accuracy: {accuracy_score(y_test, torch.max(outputs.data, 1).cpu().numpy()) * 100:.2f}%")


# --- Final Evaluation ---
print("\n--- Testing the CNN ---")
model.eval() # set to evaluation mode (turns off dropout)

with torch.no_grad(): # don't track math during testing to save memory
    inputs = X_test_tensor.to(device)
    outputs = model(inputs)
    
    # get the class with the highest predicted score
    _, predicted = torch.max(outputs.data, 1) 
    all_predictions = predicted.cpu().numpy()

# calculate Accuracy
cnn_accuracy = accuracy_score(y_test, all_predictions)
print(f"ULTIMATE PyTorch CNN ACCURACY: {cnn_accuracy * 100:.2f}%")
# --- Testing the CNN ---
# ULTIMATE PyTorch CNN ACCURACY: 91.08%