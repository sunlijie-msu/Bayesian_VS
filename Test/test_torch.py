import torch

# Create a neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Define the layers of the network
        self.fc1 = torch.nn.Linear(3, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        # Pass the input through the layers of the network
        x = self.fc1(x)
        x = self.fc2(x)

        # Return the output of the network
        return x

# Create an instance of the network
net = Net()

# Create some input data
x = torch.tensor([1, 2, 3])

# Pass the input data through the network
y = net(x)

# Print the output of the network
print(y)