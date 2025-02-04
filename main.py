import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np 

# Definir se usará GPU ou CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

hyperparams_list = [
    {"batch_size": 32, "epochs": 30, "lr": 0.001},
    {"batch_size": 64, "epochs": 35, "lr": 0.0005},
    {"batch_size": 128, "epochs": 40, "lr": 0.0001}
]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Inversão horizontal aleatória
    transforms.RandomRotation(15),  # Rotação aleatória de até 15 graus
    transforms.RandomCrop(32, padding=4),  # Recorte aleatório com padding
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Ajuste de cor
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Apenas normalização para o conjunto de teste
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Definição da CNN
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

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetModel, self).__init__()
        # Carregar modelo pré-treinado
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Substituir a última camada para corresponder ao número de classes
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.efficientnet(x)


# Função para treinar e avaliar a cada época
def train_and_evaluate(model, trainloader, testloader, criterion, optimizer, epochs, config_id):
    model.train()
    accuracy_per_epoch = []  # Lista para armazenar acurácia por época
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Avaliação do modelo após cada época
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_accuracy = 100 * correct / total
        accuracy_per_epoch.append(epoch_accuracy)

        print(f"Config {config_id} - Época [{epoch+1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}, Acurácia: {epoch_accuracy:.2f}%")

        model.train()  # Voltar ao modo de treino para a próxima época

    return accuracy_per_epoch


trainset = torchvision.datasets.CIFAR10(root="./dataset/vegetable-images/train", train=True, download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root="./dataset/vegetable-images/test", train=False, download=True, transform=test_transform)
# Executar treinamento para cada conjunto de hiperparâmetros
for i, params in enumerate(hyperparams_list, start=1):
    print(f"\nTreinando modelo {i} com hiperparâmetros: {params}")

    # Criar novo modelo
    model = EfficientNetModel().to(device)

    # Criar DataLoader com tamanho de batch específico
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"], shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=params["batch_size"], shuffle=False)

    # Criar função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    # Treinar e avaliar o modelo
    accuracies = train_and_evaluate(model, trainloader, testloader, criterion, optimizer, params["epochs"], config_id=i)

    # Calcular estatísticas de acurácia
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    # Exibir estatísticas
    print("\n--- Estatísticas para Configuração", i, "---")
    print(f"Média da acurácia ao longo das épocas: {mean_accuracy:.2f}%")
    print(f"Desvio padrão da acurácia: {std_accuracy:.2f}%")

    # Salvar o modelo treinado
    model_path = f"./models-trained/cnn_model_config{i}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo salvo como {model_path}.\n")