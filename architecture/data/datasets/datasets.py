import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Data loader
def dataloader(batch_size):

    transform = T.Compose(T.RandomCrop(32, padding=4),
                          T.RandomHorizontalFlip(),
                          T.ToTensor(),
                          T.Normalize(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    test_transform = T.Compose(T.ToTensor(),
                               T.Normalize(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    trainset = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', download=True, train=False, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader