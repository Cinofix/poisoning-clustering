import sys

sys.path.extend(["./"])

from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

# import pretrainedmodels
# import pretrainedmodels.utils as utils

# Hyperparameters
num_epochs = 50
batch_size = 128
learning_rate = 0.001


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(1024, 200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 1
        layers += [
            nn.Conv2d(in_channels, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ]
        layers += [nn.Conv2d(32, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def predict(self, image):
        self.eval()
        image = torch.clamp(image, 0, 1)
        image = Variable(image, volatile=True).view(1, 1, 28, 28)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]

    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image, 0, 1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict


def load_mnist_data():
    """ Load MNIST data from torchvision.datasets
        input: None
        output: minibatches of train and test sets
    """
    # MNIST Dataset
    train_dataset = dsets.FashionMNIST(
        root="./data/", train=True, transform=transforms.ToTensor(), download=False,
    )
    test_dataset = dsets.FashionMNIST(
        root="./data/", train=False, transform=transforms.ToTensor(), download=False,
    )

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=1000, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=10, shuffle=False
    )

    return train_loader, test_loader, train_dataset, test_dataset


def train_mnist(model, train_loader):
    # Loss and Optimizer
    model.train()
    lr = 0.01
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, nesterov=True
    )
    # Train the Model
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            images = Variable(images)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [%d/%d], Iter [%d] Loss: %.4f"
                    % (epoch + 1, num_epochs, i + 1, loss.data[0])
                )


def test_mnist(model, test_loader):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print(
        "Test Accuracy of the model on the 10000 test images: %.2f %%"
        % (100.0 * correct / total)
    )


def save_model(model, filename):
    """ Save the trained model """
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    """ Load the training model """
    model.load_state_dict(torch.load(filename))


if __name__ == "__main__":
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    net = MNIST()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    train_mnist(net, train_loader)
    test_mnist(net, test_loader)
    save_model(net, "./experiments/classifiers/cw_mnist_fashion.pt")


from sklearn.metrics import accuracy_score as ACC


class CWDeepNet:
    def __init__(self, model, name=""):
        self.model = model
        self.name = name

    @classmethod
    def load(cls, filename):
        # Load from file
        net = MNIST()
        if torch.cuda.is_available():
            net.cuda()
        load_model(net, filename)
        net = net.eval()
        return cls(net)

    def predict(self, X):
        n, m, k = X.shape
        outputs = self.model(X.view(n, 1, 28, 28))
        _, predicted = torch.max(outputs.data, 1)
        return predicted

    def score(self, X, Y):
        Y_hat = self.predict(X)
        return ACC(Y_hat.cpu(), Y.cpu())

    def to_string(self):
        return "CWDeepNet" + self.name

    def store(self, filename):
        """ Save the trained model """
        torch.save(self.model.state_dict(), filename)
