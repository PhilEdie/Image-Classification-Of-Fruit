import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import torch.nn as nn
from ConvNet import ConvNet

class Test():

    def load_test_dataset(self):
        """Loads the training images from /testdata. Applies transformations then returns a DataLoader."""
        test_data_path = "testdata"
        root = Path(test_data_path)
        classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
        print(classes)

        transform = transforms.Compose([
            transforms.Resize(300),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(250),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6071, 0.4828, 0.3934], std=[0.2845, 0.3187, 0.3240])
        ])

        dataset = datasets.ImageFolder(test_data_path, transform=transform)
        testloader = DataLoader(dataset, batch_size=4, shuffle=True)
        print("Loaded data")
        return testloader

    def test_model(self, model, dataloader, test_loss_function):
        """
        Makes predictions on the loaded test dataset using the model.pth produced by train.py.
        Returns the loss and the total correct predictions.
        """
        test_loss = 0.0
        test_correct = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss = test_loss_function(output, labels)
            test_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            test_correct += (predictions == labels).sum().item()

        return test_loss, test_correct


if __name__ == '__main__':
    # Fix the results so they are reproducible.
    torch.manual_seed(42)
    np.random.seed(42)
    # Set the device to be what ever is available. (cuda is the gpu).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    test = Test()
    # Load in the test data.
    test_loader = test.load_test_dataset()

    # Initialise the model.
    model = ConvNet()

    # Set the model to be the model created by train.py.
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    # Calculate the loss and the total correct predictions.
    test_loss, test_correct = test.test_model(model, test_loader, nn.CrossEntropyLoss())

    # Convert the loss and correct to ratios.
    test_loss = test_loss / len(test_loader.sampler)
    test_acc = (test_correct / len(test_loader.sampler)) * 100

    # Print the results.
    print("Test loss =", test_loss)
    print("Test accuracy =", test_acc)



