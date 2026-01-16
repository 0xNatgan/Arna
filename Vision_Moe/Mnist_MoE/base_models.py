import torch.nn as nn

class MNIST_CNN1(nn.Module):   
    def __init__(self):
        super(MNIST_CNN1, self).__init__()
        # layers : conv1 (k3) -> MaxPool -> conv2 (k3) -> linear 64-128 -> linear2 128-10 -> ReLU(output)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 output classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MNIST_CNN2(nn.Module):
    def __init__(self):
        super(MNIST_CNN2, self).__init__()
        # Enhanced layers : conv1 (k5) -> BN -> MaxPool -> conv2 (k5) -> BN -> linear -> Dropout -> linear2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10) # 10 output classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 128 * 7 * 7) # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MNIST_CNN3(nn.Module):
    def __init__(self):
        super(MNIST_CNN3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=2, dilation=2) # Spaced out
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Sequential(nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.view(x.size(0), -1))

    
class MNIST_MediumCNN(nn.Module):
    def __init__(self):
        super(MNIST_MediumCNN, self).__init__()
        # Enhanced architecture with double convolution blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # After 3 pools (28 -> 14 -> 7 -> 3)
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.fc_layers(x)
        return x