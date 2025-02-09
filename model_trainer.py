import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class ModelLoader:
    def __init__(self, model_path, num_classes=15, device=None):
        """
        Inicializa o carregador de modelo.
        
        Args:
            model_path: Caminho para o arquivo .pth do modelo
            num_classes: Número de classes do modelo
            device: Dispositivo para inferência (cuda/cpu)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path, num_classes)
        self.transform = self._get_transform()
        self.classes = self._get_classes()
        
    def _load_model(self, model_path, num_classes):
        """Carrega o modelo treinado"""
        # Inicializar modelo com a arquitetura correta
        model = models.inception_v3(pretrained=False)
        
        # Modificar a última camada para o número correto de classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if hasattr(model, 'AuxLogits'):
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
            
        # Carregar os pesos treinados
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Se o checkpoint contiver todo o estado do treinamento
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        model.eval()  # Colocar o modelo em modo de avaliação
        return model
    
    def _get_transform(self):
        """Define as transformações para as imagens de entrada"""
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def _get_classes(self):
        """Define as classes do seu dataset"""
        return [
            'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
            'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
            'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
        ]
    
    def predict_image(self, image_path):
        """
        Faz a predição para uma única imagem.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            tuple: (classe predita, probabilidade, todas as probabilidades)
        """
        # Carregar e transformar a imagem
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Fazer a predição
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Obter a classe com maior probabilidade
            _, predicted = torch.max(outputs, 1)
            predicted_class = self.classes[predicted.item()]
            probability = probabilities[0][predicted.item()].item()
            
            return predicted_class, probability, probabilities[0].cpu().numpy()
    
    def predict_batch(self, image_folder):
        """
        Faz predições para todas as imagens em uma pasta.
        
        Args:
            image_folder: Caminho para a pasta com imagens
            
        Returns:
            list: Lista de tuplas (nome_arquivo, classe_predita, probabilidade)
        """
        results = []
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                predicted_class, probability, _ = self.predict_image(image_path)
                results.append((filename, predicted_class, probability))
        return results

# Exemplo de uso
def main():
    # Inicializar o carregador de modelo
    model_loader = ModelLoader(
        model_path="./models-trained/inception_model_config1.pth",
        num_classes=15
    )
    
    # Exemplo de predição única
    image_path = "caminho/para/sua/imagem.jpg"
    predicted_class, probability, all_probs = model_loader.predict_image(image_path)
    print(f"\nPredição para {image_path}:")
    print(f"Classe: {predicted_class}")
    print(f"Probabilidade: {probability:.2%}")
    
    # Mostrar top 3 predições
    top3_indices = all_probs.argsort()[-3:][::-1]
    print("\nTop 3 predições:")
    for idx in top3_indices:
        print(f"{model_loader.classes[idx]}: {all_probs[idx]:.2%}")
    
    # Exemplo de predição em lote
    folder_path = "caminho/para/pasta/de/imagens"
    results = model_loader.predict_batch(folder_path)
    print("\nResultados do lote:")
    for filename, pred_class, prob in results:
        print(f"{filename}: {pred_class} ({prob:.2%})")

if __name__ == "__main__":
    main()