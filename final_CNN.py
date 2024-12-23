import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mnist_loader_10
import numpy as np
import matplotlib.pyplot as plt

# Define Dataset class
class NumericDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(np.array(inputs), dtype=torch.float32).view(-1, 3, 32, 32)
        if isinstance(labels[0], (int, np.integer)):
            self.labels = torch.tensor(labels, dtype=torch.long).squeeze()
        else:
            self.labels = torch.tensor(np.argmax(np.array(labels), axis=1), dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# Function to reduce dataset size by 10%
def reduce_dataset_size(inputs, labels, percentage=0.1):
    num_samples = len(inputs)
    num_selected = int(num_samples * percentage)
    selected_indices = np.random.choice(num_samples, num_selected, replace=False)
    reduced_inputs = [inputs[i] for i in selected_indices]
    reduced_labels = [labels[i] for i in selected_indices]
    return reduced_inputs, reduced_labels


# Load data
training_data, validation_data, test_data = mnist_loader_10.load_data_wrapper()

# Unpack data
train_inputs, train_labels = zip(*training_data)
val_inputs, val_labels = zip(*validation_data)
test_inputs, test_labels = zip(*test_data)

# Reduce the dataset size to 10% of the original
train_inputs, train_labels = reduce_dataset_size(train_inputs, train_labels)
val_inputs, val_labels = reduce_dataset_size(val_inputs, val_labels)
test_inputs, test_labels = reduce_dataset_size(test_inputs, test_labels)

# Create Dataset and DataLoader
train_dataset = NumericDataset(train_inputs, train_labels)
val_dataset = NumericDataset(val_inputs, val_labels)
test_dataset = NumericDataset(test_inputs, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)  # Output 10 classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x




# Initialize model
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)


## Modify the train_model function to record metrics
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs=30):
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_dataloader:
            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_loss / len(train_dataloader)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validate model
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                labels = labels.squeeze()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return train_accuracies, val_accuracies

# Test model
def test_model(model, test_dataloader):
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            labels = labels.squeeze()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    avg_test_loss = test_loss / len(test_dataloader)
    test_accuracy = 100 * correct_test / total_test
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Run the training and testing
train_accuracies, val_accuracies = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs=30)
test_model(model, test_dataloader)


# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, 31), val_accuracies, label='Validation Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlim([1,30])
plt.ylim([0,100])
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig(r"C:\大二上\大統\期末報告\accuracy.png",dpi = 400)