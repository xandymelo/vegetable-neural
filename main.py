import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Definir se usará GPU ou CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hiperparâmetros
# BATCH_SIZE = 64
# EPOCHS = 10
# LEARNING_RATE = 0.001

# # Transformações e carregamento do dataset CIFAR-10
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# # Definição da CNN
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 8 * 8, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 8 * 8)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# # Criar o modelo e enviar para o dispositivo (CPU/GPU)
# model = CNN().to(device)

# # Definir função de perda e otimizador
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # Função para treinar o modelo
# def train():
#     model.train()
#     for epoch in range(EPOCHS):
#         running_loss = 0.0
#         for images, labels in trainloader:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
        
#         print(f"Época [{epoch+1}/{EPOCHS}], Loss: {running_loss / len(trainloader):.4f}")

# # Função para avaliar o modelo
# def test():
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
    
#     print(f"Acurácia no conjunto de teste: {100 * correct / total:.2f}%")

# # Treinar e testar o modelo
# train()
# test()

# # Salvar o modelo treinado
# torch.save(model.state_dict(), "cnn_model.pth")
# print("Modelo salvo como cnn_model.pth")
