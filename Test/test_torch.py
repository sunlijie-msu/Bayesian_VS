import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer
        self.relu = nn.ReLU()        # Activation function
        self.fc2 = nn.Linear(10, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleNet()

# Define loss function and optimizer
# Stochastic gradient descent (SGD) as its optimization algorithm
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example data: input (features) and output (target)
features = torch.tensor([[1.0], [2.0], [3.0]])  # Example input features
targets = torch.tensor([[2.0], [4.0], [6.0]])   # Corresponding targets

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()   # Clear gradients
    outputs = model(features)  # Forward pass: compute the output
    loss = criterion(outputs, targets)  # Compute the loss
    loss.backward()  # Backward pass: compute the gradient
    optimizer.step()  # Update parameters

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    test_feature = torch.tensor([[4.0]])
    predicted = model(test_feature)
    print(f'Prediction for input 4.0: {predicted.item():.4f}')

