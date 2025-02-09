# from config import device
# import torch
# def train_and_evaluate(model, trainloader, testloader, criterion, optimizer, epochs, config_id):
#     model.train()
#     accuracy_per_epoch = []  # Lista para armazenar acurácia por época
    
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for images, labels in trainloader:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
        
#         # Avaliação do modelo após cada época
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in testloader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
        
#         epoch_accuracy = 100 * correct / total
#         accuracy_per_epoch.append(epoch_accuracy)
#         print(f"Config {config_id} - Época [{epoch+1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}, Acurácia: {epoch_accuracy:.2f}%")
#         model.train()
#     return accuracy_per_epoch


from config import device
import torch
import time
from tqdm import tqdm

def train_and_evaluate(model, trainloader, testloader, criterion, optimizer, epochs, config_id):
    """
    Treina e avalia o modelo InceptionV3, lidando com auxiliary outputs.
    
    Args:
        model: Modelo InceptionV3
        trainloader: DataLoader para dados de treino
        testloader: DataLoader para dados de teste
        criterion: Função de perda
        optimizer: Otimizador
        epochs: Número de épocas
        config_id: ID da configuração atual
    
    Returns:
        list: Lista com acurácias de teste para cada época
    """
    accuracy_per_epoch = []
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Modo de treino
        model.train()
        running_loss = 0.0
        
        # Barra de progresso para o treinamento
        train_pbar = tqdm(trainloader, desc=f'Época {epoch+1}/{epochs} [Treino]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - InceptionV3 retorna (outputs, aux_outputs) quando training=True
            if model.training and hasattr(model, 'aux_logits') and model.aux_logits:
                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2  # Peso de 0.4 para auxiliary loss
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Atualizar barra de progresso
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Modo de avaliação
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        # Barra de progresso para a avaliação
        test_pbar = tqdm(testloader, desc=f'Época {epoch+1}/{epochs} [Teste]')
        with torch.no_grad():
            for images, labels in test_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)  # Durante eval(), retorna apenas outputs
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Atualizar barra de progresso
                current_accuracy = 100 * correct / total
                test_pbar.set_postfix({'acc': f'{current_accuracy:.2f}%'})
        
        # Calcular métricas da época
        epoch_accuracy = 100 * correct / total
        epoch_train_loss = running_loss / len(trainloader)
        epoch_test_loss = test_loss / len(testloader)
        epoch_time = time.time() - epoch_start_time
        
        # Armazenar e mostrar resultados
        accuracy_per_epoch.append(epoch_accuracy)
        
        # Atualizar melhor acurácia
        is_best = epoch_accuracy > best_accuracy
        if is_best:
            best_accuracy = epoch_accuracy
        
        # Imprimir resumo da época
        print(f"\nConfig {config_id} - Época [{epoch+1}/{epochs}]")
        print(f"Tempo: {epoch_time:.1f}s")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Test Loss: {epoch_test_loss:.4f}")
        print(f"Acurácia: {epoch_accuracy:.2f}% {'(Melhor!)' if is_best else ''}")
        print("-" * 60)
        
        # Voltar para modo de treino
        model.train()
    
    return accuracy_per_epoch