import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train_and_evaluate import train_and_evaluate
from train_and_show_results import train_and_show_results
from config import hyperparams_list, trainset, testset, device
import os
import numpy as np


class CustomCNN(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(CustomCNN, self).__init__()
        # Reduzindo o número de filtros e adicionando mais pooling
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  # Reduzido de 64 para 32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Reduzido de 128 para 64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Reduzido de 256 para 128
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Reduzido de 512 para 256
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Reduzindo o tamanho das camadas fully connected
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 18 * 18, 1024),  # Reduzido de 4096 para 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),  # Reduzido de 1024 para 512
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(
    model_name="cnn",
    model_type="custom",
    save_dir="./models-trained",
    num_classes=15,
    input_channels=3,
    momentum=0.9,
    weight_decay=0.0001,
    hyperparams_list=None,
    trainset=None,
    testset=None,
    device=None
):
    os.makedirs(save_dir, exist_ok=True)
    
    # Verificar dimensões das imagens
    trainImage, label = trainset[0]
    testImage, labelImage = testset[0]
    print(f"Dimensões da imagem de treino: {trainImage.shape}")
    print(f"Dimensões da imagem de teste: {testImage.shape}")
    
    if model_type == "custom":
        model = CustomCNN(num_classes, input_channels).to(device)
    else:
        raise ValueError(f"Tipo de modelo '{model_type}' não suportado")
    
    results = []
    
    for i, params in enumerate(hyperparams_list, start=1):
        print(f"\nTreinando modelo {i} com hiperparâmetros: {params}")
        
        # Reduzindo o batch size e número de workers
        trainloader = DataLoader(
            trainset,
            batch_size=min(params["batch_size"], 32),  # Limitando batch size
            shuffle=True,
            num_workers=2,  # Reduzido de 4 para 2
            pin_memory=True  # Adiciona pin_memory para melhor performance
        )
        testloader = DataLoader(
            testset,
            batch_size=min(params["batch_size"], 32),
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Limpando cache CUDA periodicamente
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=params["lr"],
            momentum=momentum, #0.9
            weight_decay=weight_decay # 0.0001
        )
        
        accuracies = train_and_evaluate(
            model,
            trainloader,
            testloader,
            criterion,
            optimizer,
            params["epochs"],
            config_id=i
        )
        
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # Criar nomes dos arquivos
        log_path = os.path.join(save_dir, f"{model_name}_log_config{i}.txt")
        model_path = os.path.join(save_dir, f"{model_name}_model_config{i}.pth")
        
        # Preparar resultados
        results_dict = {
            "config_id": i,
            "hyperparameters": {
                **params,
                "momentum": momentum,
                "weight_decay": weight_decay
            },
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "model_path": model_path,
        }

        # Salvar modelo e logs
        with open(log_path, 'w') as log_file:
            stats_header = f"\n--- Estatísticas para Configuração {i} ---"
            accuracy_msg = f"Média da acurácia ao longo das épocas: {mean_accuracy:.2f}%"
            std_msg = f"Desvio padrão da acurácia: {std_accuracy:.2f}%"
            
            # Salvar modelo
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracies': accuracies,
                'hyperparameters': results_dict["hyperparameters"]
            }, model_path)
            
            model_msg = f"Modelo salvo como {model_path}.\n"
            
            # Exibir e salvar resultados
            print(stats_header)
            print(accuracy_msg)
            print(std_msg)
            print(model_msg)
            
            log_file.write(stats_header + '\n')
            log_file.write(accuracy_msg + '\n')
            log_file.write(std_msg + '\n')
            log_file.write(model_msg + '\n')
            
            # Detalhes adicionais
            log_file.write("\n--- Resultados Detalhados ---\n")
            log_file.write(f"ID da Configuração: {results_dict['config_id']}\n")
            log_file.write("\nHiperparâmetros:\n")
            for param_name, param_value in results_dict['hyperparameters'].items():
                log_file.write(f"{param_name}: {param_value}\n")
            log_file.write(f"\nMédia da Acurácia: {results_dict['mean_accuracy']:.2f}%\n")
            log_file.write(f"Desvio Padrão da Acurácia: {results_dict['std_accuracy']:.2f}%\n")
            log_file.write(f"Caminho do Modelo: {results_dict['model_path']}\n")

        results.append(results_dict)
    
    return results

def main():
    # Exemplo de uso com parâmetros personalizados
    results = train_model(
        model_name="cnn",
        model_type="custom",
        save_dir="./models-trained",
        num_classes=15,
        input_channels=3,
        momentum=0.9,
        weight_decay=0.0001,
        hyperparams_list=hyperparams_list,
        trainset=trainset,
        testset=testset,
        device=device
    )
    return results

if __name__ == "__main__":
    results = main()
