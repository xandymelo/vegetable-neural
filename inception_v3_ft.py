import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from train_and_evaluate import train_and_evaluate
from train_and_show_results import train_and_show_results
from config import hyperparams_list, trainset, testset, device
import os

def setup_inception_model(num_classes, device):
    """
    Configura o modelo InceptionV3 para fine-tuning
    """
    # Inicializar o modelo InceptionV3 pré-treinado
    model = models.inception_v3(pretrained=True)
    
    # Congelar as camadas iniciais
    for param in model.parameters():
        param.requires_grad = False
    
    # Descongelar e modificar as camadas finais
    # Auxiliary classifier
    if hasattr(model, 'AuxLogits'):
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        for param in model.AuxLogits.parameters():
            param.requires_grad = True
    
    # Classificador principal
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model.to(device)

def main():
    # Verificar dimensões das imagens
    trainImage, label = trainset[0]
    testImage, labelImage = testset[0]
    print(f"Dimensões da imagem de treino: {trainImage.shape}")
    print(f"Dimensões da imagem de teste: {testImage.shape}")
    
    # Configurações
    num_classes = 15
    save_dir = "./models-trained"
    os.makedirs(save_dir, exist_ok=True)
    
    # Configurar modelo
    model = setup_inception_model(num_classes, device)
    
    # Lista para armazenar resultados
    results = []
    
    # Treinar para cada conjunto de hiperparâmetros
    for i, params in enumerate(hyperparams_list, start=1):
        print(f"\nTreinando modelo {i} com hiperparâmetros: {params}")
        
        # Criar DataLoaders
        trainloader = DataLoader(
            trainset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=4
        )
        testloader = DataLoader(
            testset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=4
        )
        
        # Configurar critério e otimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([
            {'params': model.fc.parameters()},
            {'params': model.AuxLogits.parameters() if hasattr(model, 'AuxLogits') else []}
        ], lr=params["lr"])
        
        # Treinar e avaliar
        accuracies = train_and_evaluate(
            model,
            trainloader,
            testloader,
            criterion,
            optimizer,
            params["epochs"],
            config_id=i
        )
        
        # Calcular estatísticas
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        # Exibir estatísticas
        print("\n--- Estatísticas para Configuração", i, "---")
        print(f"Média da acurácia ao longo das épocas: {mean_accuracy:.2f}%")
        print(f"Desvio padrão da acurácia: {std_accuracy:.2f}%")
        
        # Salvar modelo
        model_path = os.path.join(save_dir, f"inception_model_config{i}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracies': accuracies,
            'hyperparameters': params
        }, model_path)
        print(f"Modelo salvo como {model_path}.\n")
        
        # Armazenar resultados
        results.append({
            "config_id": i,
            "hyperparameters": params,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "model_path": model_path,
            "accuracies": accuracies
        })
    
    return results

if __name__ == "__main__":
    trainImage, label = trainset[0]
    testImage, labelImage = testset[0]
    print(f"Dimensões da imagem de treino: {trainImage.shape}")
    print(f"Dimensões da imagem de teste: {trainImage.shape}")
    results = main()