import sys

sys.path.extend(["./"])

from src.classification.cifar_features.pytorchcifar.models import ResNet18
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_test_data(data_dir):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    # define transform
    transform = transforms.Compose([transforms.ToTensor(), normalize,])

    testdataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform,
    )
    return testdataset


def get_test_loader(
    data_dir, batch_size=128, shuffle=True, num_workers=4, pin_memory=False
):
    dataset = get_test_data(data_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return data_loader


def convert_keys(net_keys):
    state_dict = net_keys
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if "module." in k:
            name = k[7:]  # remove module.
        new_state_dict[name] = v
    return new_state_dict


def get_embedding(net, loader, device="cpu"):
    for data, labels in loader:
        X = data
        Y = labels
    net = net.to(device)
    net = net.eval()
    X = X.to(device)
    Y = Y.to(device)
    fx = net.get_features(X)
    return fx, X, Y


def main():
    testloader = get_test_loader(data_dir="./data/", batch_size=10000)
    net = ResNet18()
    checkpoint = torch.load(
        "./src/classification/cifar_features/pytorchcifar/checkpoint/ckpt_resnet18.pth"
    )
    weights = convert_keys(checkpoint["net"])
    net.load_state_dict(weights)
    fx, X, Y = get_embedding(net, testloader)
    torch.save(fx, "src/classification/cifar_features/cifar_features_resnet18.pt")
    torch.save(Y, "src/classification/cifar_features/cifar_labels.pt")


if __name__ == "__main__":
    main()
