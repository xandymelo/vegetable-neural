import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

from model_trainer import ModelLoader

def evaluate_model_on_folder(model_loader, test_folder, true_labels_dict=None):
    """
    Avalia o modelo em uma pasta de imagens e gera métricas de desempenho.
    
    Args:
        model_loader: Instância de ModelLoader
        test_folder: Pasta contendo as imagens de teste
        true_labels_dict: Dicionário com {nome_arquivo: classe_verdadeira}
    """
    predictions = defaultdict(list)
    true_labels = []
    pred_labels = []
    
    # Fazer predições para todas as imagens
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_folder, filename)
            predicted_class, probability, all_probs = model_loader.predict_image(image_path)
            
            # Se temos as labels verdadeiras
            if true_labels_dict and filename in true_labels_dict:
                true_class = true_labels_dict[filename]
                true_labels.append(model_loader.classes.index(true_class))
                pred_labels.append(model_loader.classes.index(predicted_class))
            
            predictions[predicted_class].append(probability)
    
    return predictions, true_labels, pred_labels

def plot_prediction_distribution(predictions, title="Distribuição de Predições por Classe"):
    """Gera gráfico de distribuição das predições por classe"""
    plt.figure(figsize=(15, 6))
    
    # Preparar dados
    classes = list(predictions.keys())
    counts = [len(predictions[c]) for c in classes]
    
    # Criar barras
    bars = plt.bar(classes, counts)
    
    # Personalizar gráfico
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Número de Predições")
    plt.xticks(rotation=45, ha='right')
    
    # Adicionar valores sobre as barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_confidence_distribution(predictions):
    """Gera boxplot da distribuição de confiança por classe"""
    plt.figure(figsize=(15, 6))
    
    # Preparar dados para o boxplot
    data = []
    labels = []
    for classe, probs in predictions.items():
        data.append(probs)
        labels.append(classe)
    
    # Criar boxplot
    plt.boxplot(data, labels=labels)
    plt.title("Distribuição de Confiança por Classe")
    plt.xlabel("Classes")
    plt.ylabel("Confiança")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, pred_labels, classes):
    """Gera matriz de confusão"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.xlabel('Predição')
    plt.ylabel('Valor Real')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    # Inicializar o modelo
    model_loader = ModelLoader(
        model_path="./models-trained/inception_model_config3.pth",
        num_classes=15
    )
    
    # Pasta com as imagens de teste
    test_folder = "./dataset/vegetable-images/train"
    
    # Se você tiver um dicionário com as labels verdadeiras
    # true_labels_dict = {
    #     "imagem1.jpg": "Tomato",
    #     "imagem2.jpg": "Potato",
    #     # ...
    # }
    
    # Fazer avaliação
    predictions, true_labels, pred_labels = evaluate_model_on_folder(
        model_loader, 
        test_folder,
        true_labels_dict=None  # Adicione seu dicionário aqui se tiver
    )
    
    # Gerar visualizações
    plot_prediction_distribution(predictions)
    plot_confidence_distribution(predictions)
    
    # Se tiver as labels verdadeiras, gerar matriz de confusão
    if true_labels:
        plot_confusion_matrix(true_labels, pred_labels, model_loader.classes)
        
        # Imprimir relatório de classificação
        print("\nRelatório de Classificação:")
        print(classification_report(
            true_labels, 
            pred_labels, 
            target_names=model_loader.classes
        ))

if __name__ == "__main__":
    main()