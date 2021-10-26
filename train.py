import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import torch.optim as optim
import torch.nn as nn
from ConvNet import ConvNet

class Train:

    def load_train_dataset(self):
        """Load the training data from /traindata. Apply the below transformations and return a DataLoader."""
        train_data_path = "traindata"
        root = Path(train_data_path)
        classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

        transform = transforms.Compose([
            transforms.Resize(300),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(250),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6071, 0.4828, 0.3934], std=[0.2845, 0.3187, 0.3240])
        ])

        dataset = datasets.ImageFolder(train_data_path, transform=transform)
        return dataset

    def train_epoch(self, model, train_loader, train_loss_function, optimizer):
        """
        Train the data for a single epoch. This function can be called multiple times to further train
        the model.
        """

        # Reset the stats.
        train_loss = 0.0
        train_correct = 0
        model.train()

        # Loop through each image within the train loader.
        for images, labels in train_loader:
            # Use whatever device is available for predictions.
            images, labels = images.to(device), labels.to(device)
            # Run the images through the ConvNet.
            output = model(images)

            # Calculate the loss using the provided loss function.
            loss = train_loss_function(output, labels)

            # Reset the gradient, run back propagation and increment the optimizer.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate the current loss and the total correct then return the results.
            train_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            train_correct += (predictions == labels).sum().item()

        return train_loss, train_correct

    def train_model(self, dataset):
        """Train the model using the provided dataset over 20 epochs. Save the trained model to a file named model.pth."""
        trainloader = DataLoader(dataset, batch_size=4, shuffle = True)
        model = ConvNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        for epoch in range(20):
            self.train_epoch(model, trainloader, nn.CrossEntropyLoss(), optimizer)

        torch.save(model.state_dict(), "./model.pth")


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device = ", device)
    train = Train()
    dataset = train.load_train_dataset()
    train.train_model(dataset)


