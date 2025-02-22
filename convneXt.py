import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.models import convnext_tiny
from torch.utils.data import DataLoader
from config import hyperparams_list, trainset, testset, device
from train_and_evaluate import train_and_evaluate

# Verificar se GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

def train_model(
    model_name="convnext",
    save_dir="./models-trained",
    num_classes=15,
    weight_decay=0.0001,
    hyperparams_list=None,
    trainset=None,
    testset=None,
    device=None
):
    os.makedirs(save_dir, exist_ok=True)

    results = []

    for i, params in enumerate(hyperparams_list, start=1):
        print(f"\nTreinando modelo {i} com hiperparâmetros: {params}")

        # Criando DataLoaders com menor batch size
        batch_size = min(params["batch_size"], 16)  # Reduzindo para evitar estouro de memória
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Reduzindo para evitar consumo excessivo de RAM
            pin_memory=True
        )
        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        # Limpar cache da GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Carregar modelo
        model = convnext_tiny(weights="IMAGENET1K_V1")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        model = model.to(device)

        # Compilar modelo para otimizar memória
        model = torch.compile(model, mode="reduce-overhead") if torch.__version__ >= "2.0.0" else model

        # Configurar loss e otimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=weight_decay)

        # Treinar e avaliar
        accuracies = train_and_evaluate(model, trainloader, testloader, criterion, optimizer, params["epochs"], config_id=i)

        # Calcular estatísticas
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # Salvar modelo e resultados
        model_path = os.path.join(save_dir, f"{model_name}_model_config{i}.pth")
        torch.save({'model_state_dict': model.state_dict()}, model_path)

        print(f"Config {i}: Média da acurácia: {mean_accuracy:.2f}% | Desvio padrão: {std_accuracy:.2f}%")

        results.append({
            "config_id": i,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "model_path": model_path,
        })

        # Liberar memória
        del model
        torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    results = train_model(
        model_name="convnext",
        save_dir="./models-trained",
        num_classes=15,
        hyperparams_list=hyperparams_list,
        trainset=trainset,
        testset=testset,
        device=device
    )
