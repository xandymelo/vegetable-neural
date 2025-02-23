import torch
import torchvision
import numpy as np
import csv
from config import testset
from inception_v3_ft import setup_inception_model

# Caminho para o arquivo do modelo treinado
model_path = 'models-trained/inception_model_config3.pth'

# Definindo o dispositivo (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurando o modelo
num_classes = 15  # Ajuste para o número correto de classes
print("Configurando o modelo...")
model = setup_inception_model(num_classes, device, model_path=model_path)
print("Modelo configurado com sucesso!")

# Colocar o modelo em modo de avaliação
model.eval()

# Função para pegar as previsões erradas
def get_incorrect_predictions(model, dataloader):
    incorrect_images = []
    incorrect_labels = []
    incorrect_preds = []
    
    with torch.no_grad():  # Desativar gradientes para economizar memória
        for i, (inputs, labels) in enumerate(dataloader):
            print(f"Processando batch {i + 1}/{len(dataloader)}...")
            # Enviar para GPU se disponível
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
                
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Pega a classe com maior probabilidade

            # Comparar as previsões com os rótulos verdadeiros
            incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in incorrect_indices:
                incorrect_images.append(inputs[idx])
                incorrect_labels.append(labels[idx].item())
                incorrect_preds.append(preds[idx].item())
    
    print(f"Total de previsões erradas: {len(incorrect_images)}")
    return incorrect_images, incorrect_labels, incorrect_preds


# Criar o DataLoader para o conjunto de teste
print("Criando o DataLoader para o conjunto de teste...")
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
print("DataLoader criado com sucesso!")

# Obter as previsões erradas
print("Obtendo previsões erradas...")
incorrect_images, incorrect_labels, incorrect_preds = get_incorrect_predictions(model, test_loader)

# Gerar um relatório com as previsões erradas
report_file = 'inception_relatorio_previsoes_erradas.csv'

# Criando o cabeçalho para o CSV
header = ['Índice', 'Rótulo Verdadeiro', 'Previsão Errada']

# Escrevendo as previsões erradas no arquivo CSV
print(f"Gerando relatório em {report_file}...")
with open(report_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for i in range(len(incorrect_images)):
        true_label = testset.classes[incorrect_labels[i]]
        pred_label = testset.classes[incorrect_preds[i]]
        writer.writerow([i, true_label, pred_label])

print(f"Relatório gerado com sucesso! Verifique o arquivo {report_file}.")
