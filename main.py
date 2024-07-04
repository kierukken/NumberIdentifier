import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

trainset = datasets.MNIST('~./data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=6000, shuffle=True)

testset = datasets.MNIST('~./data', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=True)

class Model(nn.Module):
    def __init__(self, in_features=784, h1 = 390, h2 = 195, h3 = 97, out_features = 10, dropout_rate = 0.3):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(h1, h2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(h2, h3)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(h3, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.out(x)
        return x

model = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
epochs = 10

for epoch in range(epochs):
    for images, labels in trainloader:
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images.view(-1, 28*28))
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

correct = 0
total = 0
test_loss = 0.0
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images.view(-1, 28*28))
        loss = loss_fn(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total}%')
print(f'Test Loss: {test_loss / len(testloader)}')
