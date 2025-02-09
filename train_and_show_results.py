import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os

def train_and_show_results(
    model_class,
    hyperparams_list,
    trainset,
    testset,
    device,
    file_name,
    save_dir="./models-trained",
    train_and_evaluate_fn=None,
):
    """
    Treina múltiplas configurações de um modelo com diferentes hiperparâmetros.
    
    Args:
        model_class: Classe do modelo a ser instanciado (ex: EfficientNetModel)
        hyperparams_list: Lista de dicionários com hiperparâmetros
        trainset: Dataset de treino
        testset: Dataset de teste
        device: Dispositivo para treinamento ('cuda' ou 'cpu')
        save_dir: Diretório para salvar os modelos treinados
        train_and_evaluate_fn: Função de treino e avaliação personalizada
        
    Returns:
        list: Lista de dicionários com resultados de cada configuração
    """
    
    # Criar diretório para salvar modelos se não existir
    os.makedirs(save_dir, exist_ok=True)
    
    # Lista para armazenar resultados
    results = []
    
    # Função padrão de treino e avaliação se nenhuma for fornecida
    if train_and_evaluate_fn is None:
        raise ValueError("É necessário fornecer uma função train_and_evaluate_fn")

    # Executar treinamento para cada conjunto de hiperparâmetros
    for i, params in enumerate(hyperparams_list, start=1):
        print(f"\nTreinando modelo {i} com hiperparâmetros: {params}")
        
        model = model_class().to(device)
        
        # Criar DataLoader com tamanho de batch específico
        trainloader = DataLoader(
            trainset, 
            batch_size=params["batch_size"], 
            shuffle=True
        )
        testloader = DataLoader(
            testset, 
            batch_size=params["batch_size"], 
            shuffle=False
        )
        
        # Criar função de perda e otimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        
        # Treinar e avaliar o modelo
        accuracies = train_and_evaluate_fn(
            model, 
            trainloader, 
            testloader, 
            criterion, 
            optimizer, 
            params["epochs"], 
            config_id=i
        )
        
        # Calcular estatísticas de acurácia
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        # Exibir estatísticas
        print("\n--- Estatísticas para Configuração", i, "---")
        print(f"Média da acurácia ao longo das épocas: {mean_accuracy:.2f}%")
        print(f"Desvio padrão da acurácia: {std_accuracy:.2f}%")
        
        # Salvar o modelo treinado
        model_path = os.path.join(save_dir, f"model_config{i}.pth")
        torch.save(model.state_dict(), model_path)
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