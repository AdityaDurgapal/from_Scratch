from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from main import CNN
from Layers.conv import Conv2D
from Layers.pooling import MaxPool2D
from Layers.dense import Dense
from Layers.activation import ReLU, Softmax

print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

X = X.astype('float32') / 255.0 
y = y.astype('int') 

X = X.reshape(-1, 28, 28)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Test labels: {y_test.shape}")

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

y_train_onehot = one_hot_encode(y_train)
y_test_onehot = one_hot_encode(y_test)

print(f"One-hot training labels: {y_train_onehot.shape}")
print(f"One-hot test labels: {y_test_onehot.shape}")

class CrossEntropyLoss:
    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]

        if targets.shape[0] != batch_size:
            print(f"Warning: Batch size mismatch. Predictions: {predictions.shape}, Targets: {targets.shape}")
            return float('inf')
            
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        loss = -np.sum(targets * np.log(predictions)) / batch_size
        self.predictions = predictions
        self.targets = targets
        return loss
        
    def backward(self):
        batch_size = self.predictions.shape[0]
        return (self.predictions - self.targets) / batch_size

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                weights_grad = np.clip(layer.weights_gradient, -1.0, 1.0)
                biases_grad = np.clip(layer.biases_gradient, -1.0, 1.0)
                
                layer.weights -= self.learning_rate * weights_grad
                layer.biases -= self.learning_rate * biases_grad

def create_batches(X, y, batch_size=4):
    batches = []
    for i in range(0, len(X), batch_size):
        end_idx = min(i + batch_size, len(X))
        batch_X = X[i:end_idx]
        batch_y = y[i:end_idx]

        if len(batch_X) == batch_size:
            batches.append((batch_X, batch_y))
    return batches

X_train = X_train 
y_train = y_train_onehot

X_test = X_test 
y_test = y_test_onehot

model = CNN()

model.add_layer(Conv2D(num_filters=8, filter_size=3, padding=1))
model.add_layer(ReLU())
model.add_layer(MaxPool2D(pool_size=2, stride=2))

model.add_layer(Conv2D(num_filters=16, filter_size=3, padding=1))
model.add_layer(ReLU())
model.add_layer(MaxPool2D(pool_size=2, stride=2))

model.add_layer(Dense(64)) 
model.add_layer(ReLU())
model.add_layer(Dense(10)) 
model.add_layer(Softmax())

model.compile(optimizer=SGD(learning_rate=0.01), 
              loss_function=CrossEntropyLoss())

num_epochs = 30
batch_size = 200
print("Starting training on MNIST dataset...")
for epoch in range(num_epochs):
    batches = create_batches(X_train, y_train, batch_size)
    
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Number of batches: {len(batches)}")
    
    for batch_idx, (x_batch, y_batch) in enumerate(batches):
        print(f"Processing batch {batch_idx+1}/{len(batches)}...")
        loss, predictions = model.train_step(x_batch, y_batch)

        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_batch, axis=1)
        correct_predictions += np.sum(predicted_classes == true_classes)
        total_predictions += len(true_classes)
        
        epoch_loss += loss

        if (batch_idx + 1) % 50 == 0:
            current_acc = np.sum(predicted_classes == true_classes) / len(true_classes)
            print(f"Batch {batch_idx+1}, Loss: {loss:.4f}, Batch Accuracy: {current_acc:.4f}")
        
    
    avg_loss = epoch_loss / min(len(batches), 16)
    accuracy = correct_predictions / total_predictions
    
    print(f"Epoch {epoch+1} Complete - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

print("\nTraining completed!")

print("Evaluating on test set...")
test_batches = create_batches(X_test, y_test, batch_size)
test_correct = 0
test_total = 0

for x_batch, y_batch in test_batches:
    predictions = model.predict(x_batch)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_batch, axis=1)
    test_correct += np.sum(predicted_classes == true_classes)
    test_total += len(true_classes)

test_accuracy = test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nImplementation completed successfully!")
print(f"Final Test Accuracy: {test_accuracy:.4f}")
