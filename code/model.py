import torch
import torch.nn as nn
import torch.optim as optim

# Load config file variables
from config_loader import load_config
config = load_config()
num_epochs = config['model_num_epochs']
criterion_mapping = { int(k): getattr(torch.nn, v) for k, v in config['model_criterion_mapping'].items() }
optimizer_mapping = { int(k): getattr(torch.optim, v) for k, v in config['model_optimizer_mapping'].items() }


# Particle model class
class Model(nn.Module):
    # Create CNN with given parameters
    def __init__(self, num_conv_layers, num_filters, dropout_rate):
        super(Model, self).__init__()
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(3 if i == 0 else num_filters, num_filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            ) for i in range(num_conv_layers)]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_filters * (32 // (2 ** num_conv_layers)) ** 2, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


# Manage particle model
class ModelHandler:
    # Parse through particle parameters and create model
    def __init__(self, particle: list):
        # Get particle parameters
        num_conv_layers = particle[0]
        num_filters = particle[1]
        dropout_rate = particle[2]

        self.lr = particle[3]
        self.criterion_choice = criterion_mapping[int(particle[4])]
        self.optimizer_choice = optimizer_mapping[int(particle[5])]

        # Create particle model
        self.model = Model(num_conv_layers, num_filters, dropout_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # Train model
    def train(self, train_loader):
        criterion = self.criterion_choice()
        optimizer = self.optimizer_choice(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    # Evaluate model
    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate and return accuracy
        accuracy = correct / total
        return accuracy
    
    # Count number of parameters in model
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params