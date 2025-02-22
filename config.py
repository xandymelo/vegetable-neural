import torchvision.transforms as transforms
import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hyperparams_list = [
    {"batch_size": 32, "epochs": 15, "lr": 0.001},
    {"batch_size": 64, "epochs": 15, "lr": 0.0005},
    {"batch_size": 128, "epochs": 20, "lr": 0.0001}
]

# DATA AUMENTATITION

train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(p=0.5),  # Inversão horizontal aleatória
    transforms.RandomRotation(15),  # Rotação aleatória de até 15 graus
    transforms.Resize((299, 299)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Ajuste de cor
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.ImageFolder(root="./dataset/vegetable-images/train", transform=train_transform)
testset = torchvision.datasets.ImageFolder(root="./dataset/vegetable-images/test", transform=test_transform)